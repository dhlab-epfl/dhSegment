from xml.etree import ElementTree as ET
from typing import List, Optional, Union, Tuple
import numpy as np
import datetime
import cv2

# https://docs.python.org/3.5/library/xml.etree.elementtree.html#parsing-xml-with-namespaces
_ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
_attribs = {'xmlns:xsi': "http://www.w3.org/2001/XMLSchema-instance",
            'xsi:schemaLocation': "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 "
                                  "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"}


def _try_to_int(d: Optional[Union[str, int]])-> Optional[int]:
    if isinstance(d, str):
        return int(d)
    else:
        return d


def _get_text_equiv(e: ET.Element) -> str:
    tmp = e.find('p:TextEquiv', _ns)
    if tmp is None:
        return ''
    tmp = tmp.find('p:Unicode', _ns)
    if tmp is None:
        return ''
    return tmp.text


class Point:
    def __init__(self, y: int, x: int):
        self.y = y
        self.x = x

    @classmethod
    def list_from_xml(cls, e: ET.Element) -> List['Point']:
        if e is None:
            # print('warning, trying to construct list of points from None, defaulting to []')
            return []
        t = e.attrib['points']
        result = []
        for p in t.split(' '):
            values = p.split(',')
            assert len(values) == 2
            x, y = int(float(values[0])), int(float(values[1]))
            result.append(Point(y, x))
        return result

    @classmethod
    def list_to_cv2poly(cls, list_points: List['Point']) -> np.array:
        return np.array([(p.x, p.y) for p in list_points], dtype=np.int32).reshape([-1, 1, 2])

    @classmethod
    def cv2_to_point_list(cls, cv2_array) -> List['Point']:
        return [Point(p[0, 1], p[0, 0]) for p in cv2_array]

    @classmethod
    def list_point_to_string(cls, list_points: List['Point']) -> str:
        return ' '.join(['{},{}'.format(p.x, p.y) for p in list_points])


class BaseElement:
    tag = None

    @classmethod
    def full_tag(cls) -> str:
        return '{{{}}}{}'.format(_ns['p'], cls.tag)

    @classmethod
    def check_tag(cls, tag):
        assert tag == cls.full_tag(), 'Invalid tag, expected {} got {}'.format(cls.full_tag(), tag)


class Region(BaseElement):
    tag = 'Region'

    def __init__(self, id: str=None, coords: List[Point]=None):
        self.coords = coords if coords is not None else []
        self.id = id

    @classmethod
    def from_xml(cls, e: ET.Element) -> dict:
        return {'id': e.attrib.get('id'),
                'coords': Point.list_from_xml(e.find('p:Coords', _ns))}

    def to_xml(self, name_element: str=None) -> ET.Element:
        et = ET.Element(name_element if name_element is not None else '')
        et.set('id', self.id if self.id is not None else '')
        if not not self.coords:
            coords = ET.SubElement(et, 'Coords')
            coords.set('points', Point.list_point_to_string(self.coords))
        return et


class TextLine(Region):
    tag = 'TextLine'

    def __init__(self, id: str=None, coords: List[Point]=None, baseline: List[Point]=None, text_equiv: str=''):
        super().__init__(id=id, coords=coords)
        self.baseline = baseline if baseline is not None else []
        self.text_equiv = text_equiv

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextLine':
        cls.check_tag(e.tag)
        return TextLine(
            **super().from_xml(e),
            baseline=Point.list_from_xml(e.find('p:Baseline', _ns)),
            text_equiv=_get_text_equiv(e)
        )

    @classmethod
    def from_array(cls, cv2_coords: np.array=None, baseline_coords: np.array=None,  # cv2_coords shape [N, 1, 2]
                   text_equiv: str=None, id: str=None):
        return TextLine(
            id=id,
            coords=Point.cv2_to_point_list(cv2_coords) if cv2_coords is not None else [],
            baseline=Point.cv2_to_point_list(baseline_coords) if baseline_coords is not None else [],
            text_equiv=text_equiv
        )

    def to_xml(self, name_element='TextLine') -> ET.Element:
        line_et = super().to_xml(name_element=name_element)
        if not not self.baseline:
            line_baseline = ET.SubElement(line_et, 'Baseline')
            line_baseline.set('points', Point.list_point_to_string(self.baseline))
        line_text_equiv = ET.SubElement(line_et, 'TextEquiv')
        text_unicode = ET.SubElement(line_text_equiv, 'Unicode')
        if not not self.text_equiv:
            text_unicode.text = self.text_equiv
        return line_et

    def scale_baseline_points(self, ratio):
        scaled_points = list()
        for pt in self.baseline:
            scaled_points.append(Point(int(pt.y * ratio[0]), int(pt.x * ratio[1])))

        self.baseline = scaled_points


class GraphicRegion(Region):
    """
    Regions containing simple graphics, such as a company logo, should be marked as graphic regions.
    """
    tag = 'GraphicRegion'

    def __init__(self, id: str=None, coords: List[Point]=None,):
        super().__init__(id=id, coords=coords)

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'GraphicRegion':
        cls.check_tag(e.tag)
        return GraphicRegion(**super().from_xml(e))

    def to_xml(self, name_element='GraphicRegion') -> ET.Element:
        graph_et = super().to_xml(name_element)

        return graph_et


class TextRegion(Region):
    tag = 'TextRegion'

    def __init__(self, id: str=None, coords: List[Point]=None, text_lines: List[TextLine]=None, text_equiv: str=''):
        super().__init__(id=id, coords=coords)
        self.text_equiv = text_equiv
        self.text_lines = text_lines if text_lines is not None else []

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextRegion':
        cls.check_tag(e.tag)
        return TextRegion(
            **super().from_xml(e),
            text_lines=[TextLine.from_xml(tl) for tl in e.findall('p:TextLine', _ns)],
            text_equiv=_get_text_equiv(e)
        )

    def to_xml(self, name_element='TextRegion') -> ET.Element:
        text_et = super().to_xml(name_element=name_element)
        for tl in self.text_lines:
            text_et.append(tl.to_xml())
        text_equiv = ET.SubElement(text_et, 'TextEquiv')
        text_unicode = ET.SubElement(text_equiv, 'Unicode')
        if not not self.text_equiv:
            text_unicode.text = self.text_equiv
        return text_et


class TableRegion(Region):
    """
    Tabular data in any form is represented with a table region. Rows and columns may or may not have separator
    lines; these lines are not separator regions.
    """

    tag = 'TableRegion'

    def __init__(self, id: str=None, coords: List[Point]=None, rows: int=None, columns: int=None,
                 embeded_text: bool=None):
        super().__init__(id=id, coords=coords)
        self.rows = rows
        self.columns = columns
        self.embText = embeded_text

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TableRegion':
        cls.check_tag(e.tag)
        return TableRegion(
            **super().from_xml(e),
            rows=e.attrib.get('rows'),
            columns=e.attrib.get('columns'),
            embeded_text=e.attrib.get('embText')
        )

    def to_xml(self, name_element='TableRegion') -> ET.Element:
        table_et = super().to_xml(name_element)
        table_et.set('rows', self.rows if self.rows is not None else 0)
        table_et.set('columns', self.columns if self.columns is not None else 0)
        table_et.set('embText', self.embText if self.embText is not None else False)
        return table_et


class SeparatorRegion(Region):
    """
    Separators are lines that lie between columns and paragraphs and can be used to logically separate
    different articles from each other.
    """

    tag = 'SeparatorRegion'

    def __init__(self, id: str, coords: List[Point]=None):
        super().__init__(id=id, coords=coords)

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'SeparatorRegion':
        cls.check_tag(e.tag)
        return SeparatorRegion(**super().from_xml(e))

    def to_xml(self, name_element='SeparatorRegion') -> ET.Element:
        separator_et = super().to_xml(name_element)
        return separator_et


class Border(BaseElement):
    """
    Border of the actual page (if the scanned image contains parts not belonging to the page).
    """

    tag = 'Border'

    def __init__(self, coords: List[Point]=None):
        self.coords = coords if coords is not None else []

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'Border':
        if e is None:
            return None
        cls.check_tag(e.tag)
        return Border(
            coords=Point.list_from_xml(e.find('p:Coords', _ns))
        )

    def to_xml(self) -> ET.Element:
        border_et = ET.Element('Border')
        if not not self.coords:
            border_coords = ET.SubElement(border_et, 'Coords')
            border_coords.set('points', Point.list_point_to_string(self.coords))
        return border_et


class Page(BaseElement):
    tag = 'Page'

    def __init__(self, image_filename: str=None, image_width: int=None, image_height: int=None,
                 text_regions: List[TextRegion]=None, graphic_regions: List[GraphicRegion]=None,
                 page_border: Border=None, separator_regions: List[SeparatorRegion]=None,
                 table_regions: List[TableRegion]=None):
        self.image_filename = image_filename
        self.image_width = _try_to_int(image_width)
        self.image_height = _try_to_int(image_height)
        self.text_regions = text_regions if text_regions is not None else []
        self.graphic_regions = graphic_regions if graphic_regions is not None else []
        self.border = page_border if page_border is not None else []
        self.separator_regions = separator_regions if separator_regions is not None else []
        self.table_regions = table_regions if table_regions is not None else []

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'Page':
        cls.check_tag(e.tag)
        return Page(
            image_filename=e.attrib.get('imageFilename'),
            image_width=e.attrib.get('imageWidth'),
            image_height=e.attrib.get('imageHeight'),
            text_regions=[TextRegion.from_xml(tr) for tr in e.findall('p:TextRegion', _ns)],
            graphic_regions=[GraphicRegion.from_xml(tr) for tr in e.findall('p:GraphicRegion', _ns)],
            page_border=Border.from_xml(e.find('p:Border', _ns)),
            separator_regions=[SeparatorRegion.from_xml(sep) for sep in e.findall('p:SeparatorRegion', _ns)],
            table_regions=[TableRegion.from_xml(tr) for tr in e.findall('p:TableRegion', _ns)]
        )

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'Page':
        return Page(image_filename=dictionary.get('image_filename'),
                    image_width=dictionary.get('image_width'),
                    image_height=dictionary.get('image_height'),
                    text_regions=dictionary.get('text_regions'),
                    graphic_regions=dictionary.get('graphic_regions'),
                    page_border=dictionary.get('page_border'),
                    separator_regions=dictionary.get('separator_regions'),
                    table_regions=dictionary.get('table_regions')
                    )

    def to_xml(self) -> ET.Element:
        page_et = ET.Element('Page')
        if self.image_filename:
            page_et.set('imageFilename', self.image_filename)
        if self.image_width:
            page_et.set('imageWidth', str(self.image_width))
        if self.image_height:
            page_et.set('imageHeight', str(self.image_height))
        for tr in self.text_regions:
            page_et.append(tr.to_xml())
        for gr in self.graphic_regions:
            page_et.append(gr.to_xml())
        if self.border:
            page_et.append(self.border.to_xml())
        for sep in self.separator_regions:
            page_et.append(sep.to_xml())
        for tr in self.table_regions:
            page_et.append(tr.to_xml())
        return page_et

    def write_to_file(self, filename, creator_name='dhSegment'):
        root = ET.Element('PcGts')
        root.set('xmlns', _ns['p'])
        # Metadata
        generated_on = str(datetime.datetime.now().isoformat())
        metadata = ET.SubElement(root, 'Metadata')
        creator = ET.SubElement(metadata, 'Creator')
        creator.text = creator_name
        created = ET.SubElement(metadata, 'Created')
        # TODO : Consider the case where the file already exists and only an update needs to be done
        created.text = generated_on
        last_change = ET.SubElement(metadata, 'LastChange')
        last_change.text = generated_on

        root.append(self.to_xml())
        for k, v in _attribs.items():
            root.attrib[k] = v
        ET.ElementTree(element=root).write(filename)

    def draw_baselines(self, img_canvas, color=(255, 0, 0), thickness=2, endpoint_radius=4, autoscale=True):
        """
        Given an image, draws the TextLines.baselines.
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: the thickness of the line
        :param endpoint_radius: the radius of the endpoints of line s(first and last coordinates of line)
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        text_lines = [tl for tr in self.text_regions for tl in tr.text_lines]
        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        tl_coords = [(Point.list_to_cv2poly(tl.baseline)*ratio).astype(np.int32) for tl in text_lines
                     if len(tl.baseline) > 0]
        cv2.polylines(img_canvas, tl_coords, False, color, thickness=thickness)
        for coords in tl_coords:
            cv2.circle(img_canvas, (coords[0, 0, 0], coords[0, 0, 1]),
                       radius=endpoint_radius, color=color, thickness=-1)
            cv2.circle(img_canvas, (coords[-1, 0, 0], coords[-1, 0, 1]),
                       radius=endpoint_radius, color=color, thickness=-1)

    def draw_lines(self, img_canvas, color=(255, 0, 0), thickness=2, fill: bool=True, autoscale=True):
        """
        Given an image, draws the polygons containing text lines, i.e TextLines.coords
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        text_lines = [tl for tr in self.text_regions for tl in tr.text_lines]
        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        tl_coords = [(Point.list_to_cv2poly(tl.coords)*ratio).astype(np.int32) for tl in text_lines
                     if len(tl.coords) > 0]

        if fill:
            for tl in tl_coords:  # For loop to avoid black regions when lines overlap
                cv2.fillPoly(img_canvas, [tl], color)
        else:
            for tl in tl_coords:  # For loop to avoid black regions when lines overlap
                cv2.polylines(img_canvas, [tl], False, color, thickness=thickness)

    def draw_text_regions(self, img_canvas, color: Tuple[int, int, int]=(255, 0, 0), fill: bool=True,
                          thickness: int=3, autoscale: bool=True):
        """
        Given an image, draws the TextRegions, either fills it (fill=True) or draws the contours (fill=False)
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=True the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        tr_coords = [(Point.list_to_cv2poly(tr.coords)*ratio).astype(np.int32) for tr in self.text_regions
                     if len(tr.coords) > 0]
        if fill:
            cv2.fillPoly(img_canvas, tr_coords, color)
        else:
            cv2.polylines(img_canvas, tr_coords, True, color, thickness=thickness)

    def draw_page_border(self, img_canvas, color: Tuple[int, int, int]=(255, 0, 0), fill: bool=True,
                         thickness: int=5, autoscale: bool=True):
        """
        Given an image, draws the page border, either fills it (fill=True) or draws the contours (fill=False)
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=True the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        border_coords = (Point.list_to_cv2poly(self.border.coords) * ratio).astype(np.int32) \
            if len(self.border.coords) > 0 else []
        if fill:
            cv2.fillPoly(img_canvas, [border_coords], color)
        else:
            cv2.polylines(img_canvas, [border_coords], True, color, thickness=thickness)

    def draw_separator_lines(self, img_canvas: np.array, color: Tuple[int, int, int]=(0, 255, 0),
                             thickness: int=3, filter_by_id: str='', autoscale: bool=True):
        """
        Given an image, draws the SeparatorRegion.
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: thickness of the line
        :param filter_by_id: string to filter the lines by id. For example vertical/horizontal lines can be filtered
                if 'vertical' or 'horizontal' is mentioned in the id.
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        sep_coords = [(Point.list_to_cv2poly(sep.coords) * ratio).astype(np.int32) for sep in self.separator_regions
                      if len(sep.coords) > 0 and filter_by_id in sep.id]
        cv2.polylines(img_canvas, sep_coords, True, color, thickness=thickness)

    def draw_graphic_regions(self, img_canvas, color: Tuple[int, int, int]=(255, 0, 0),
                             fill: bool=True, thickness: int=3, autoscale: bool=True):
        """
        Given an image, draws the GraphicRegions, either fills it (fill=True) or draws the contours (fill=False)
        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=True the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True, it will use the dimensions
        provided in Page.image_width and Page.image_height to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        gr_coords = [(Point.list_to_cv2poly(gr.coords)*ratio).astype(np.int32) for gr in self.graphic_regions
                     if len(gr.coords) > 0]
        if fill:
            cv2.fillPoly(img_canvas, gr_coords, color)
        else:
            cv2.polylines(img_canvas, gr_coords, True, color, thickness=thickness)


def parse_file(filename: str) -> Page:
    """

    :param filename: PAGE xml to parse
    :return: Page object containing all the parsed elements
    """
    xml_page = ET.parse(filename)
    page_elements = xml_page.findall('p:Page', _ns)
    # can there be multiple pages in a single XML file? -> I don't think so
    assert len(page_elements) == 1
    return Page.from_xml(page_elements[0])


def save_baselines(filename, baselines, ratio=(1, 1), initial_shape=None):
    txt_lines = [TextLine.from_array(baseline_coords=b, id='line_{}'.format(i)) for i, b in enumerate(baselines)]
    for l in txt_lines:
        l.scale_baseline_points(ratio)
    txt_region = TextRegion(text_lines=txt_lines, id='region_0')
    page = Page(text_regions=[txt_region],
                image_height=int(initial_shape[0]*ratio[0]) if initial_shape is not None else None,
                image_width=int(initial_shape[1]*ratio[1]) if initial_shape is not None else None)
    page.write_to_file(filename)
