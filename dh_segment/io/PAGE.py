from xml.etree import ElementTree as ET
from typing import List, Optional, Union, Tuple
import numpy as np
import datetime
import cv2
import os
import json
from uuid import uuid4
from shapely.geometry import Polygon

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

    @classmethod
    def array_to_list(cls, array: np.ndarray) -> list:
        """

        :param array: Array must be of shape (N, 2)
        :return: list of shape (N,2)
        """
        return [list(pt) for pt in array]

    @classmethod
    def list_to_point(cls, list_coords: np.ndarray) -> List['Point']:
        """

        :param list_coords: list of shape (N, 2)
        :return: list of Points
        """
        return [cls(coord[1], coord[0]) for coord in list_coords if list_coords]

    @classmethod
    def point_to_list(cls, points: List['Point']) -> list:
        """

        :param points: list of Points
        :return: list of shape (N,2)
        """
        return [[pt.x, pt.y] for pt in points]

    def to_dict(self):
        return [int(self.x), int(self.y)]


class Text:
    def __init__(self, text_equiv: str=None, alternatives: List[str]=None, score: float=None):
        self.text_equiv = text_equiv  # if text_equiv is not None else ''
        self.alternatives = alternatives  # if alternatives is not None else []
        self.score = score  # if score is not None else ''

    def to_dict(self):
        return vars(self)


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

    def to_dict(self, non_serializable_keys: List[str]=list()) -> dict:
        if 'coords' in vars(self).keys() and 'coords' not in non_serializable_keys:
            non_serializable_keys += ['coords']
        return json_serialize(vars(self), non_serializable_keys=non_serializable_keys)

    @classmethod
    def from_dict(cls, dictionary: dict) -> dict:
        return {'id': dictionary.get('id'),
                'coords': Point.list_to_point(dictionary.get('coords'))
                }


class TextLine(Region):
    tag = 'TextLine'

    # def __init__(self, id: str=None, coords: List[Point]=None, baseline: List[Point]=None, text_equiv: str=''):
    def __init__(self, id: str = None, coords: List[Point] = None, baseline: List[Point] = None, text: Text = None,
                 line_group_id: str = None, column_group_id: str = None):
        super().__init__(id=id if id is not None else str(uuid4()), coords=coords)
        self.baseline = baseline if baseline is not None else []
        # self.text_equiv = text_equiv if text_equiv is not None else ''
        self.text = text if text is not None else Text()
        self.line_group_id = line_group_id if line_group_id is not None else ''
        self.column_group_id = column_group_id if column_group_id is not None else ''

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextLine':
        cls.check_tag(e.tag)
        return TextLine(
            **super().from_xml(e),
            baseline=Point.list_from_xml(e.find('p:Baseline', _ns)),
            text=Text(text_equiv=_get_text_equiv(e))
        )

    @classmethod
    def from_array(cls, cv2_coords: np.array=None, baseline_coords: np.array=None,  # cv2_coords shape [N, 1, 2]
                   text_equiv: str=None, id: str=None):
        return TextLine(
            id=id,
            coords=Point.cv2_to_point_list(cv2_coords) if cv2_coords is not None else [],
            baseline=Point.cv2_to_point_list(baseline_coords) if baseline_coords is not None else [],
            text=Text(text_equiv=text_equiv)
        )

    def to_xml(self, name_element='TextLine') -> ET.Element:
        line_et = super().to_xml(name_element=name_element)
        if not not self.baseline:
            line_baseline = ET.SubElement(line_et, 'Baseline')
            line_baseline.set('points', Point.list_point_to_string(self.baseline))
        line_text_equiv = ET.SubElement(line_et, 'TextEquiv')
        text_unicode = ET.SubElement(line_text_equiv, 'Unicode')
        if not not self.text.text_equiv:
            text_unicode.text = self.text.text_equiv
        return line_et

    def scale_baseline_points(self, ratio):
        scaled_points = list()
        for pt in self.baseline:
            scaled_points.append(Point(int(pt.y * ratio[0]), int(pt.x * ratio[1])))

        self.baseline = scaled_points

    def to_dict(self, non_serializable_keys: List[str]=list()):
        return super().to_dict(non_serializable_keys=['text', 'baseline'])

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'TextLine':
        return cls(**super().from_dict(dictionary),
                   baseline=Point.list_to_point(dictionary.get('baseline')),
                   text=Text(**dictionary.get('text', dict())),
                   line_group_id=dictionary.get('line_group_id'),
                   column_group_id=dictionary.get('column_group_id')
                   )


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

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'GraphicRegion':
        return cls(**super().from_dict(dictionary))


class TextRegion(Region):
    tag = 'TextRegion'

    def __init__(self, id: str=None, coords: List[Point]=None, text_lines: List[TextLine]=None, text_equiv: str=''):
        super().__init__(id=id, coords=coords)
        self.text_equiv = text_equiv if text_equiv is not None else ''
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

    def to_dict(self, non_serializable_keys: List[str]=list()):
        return super().to_dict(non_serializable_keys=['text_lines'])

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'TextRegion':
        return cls(**super().from_dict(dictionary),
                   text_lines=[TextLine.from_dict(tl) for tl in dictionary.get('text_lines', list())],
                   text_equiv=dictionary.get('text_equiv')
                   )


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

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'TableRegion':
        return cls(**super().from_dict(dictionary),
                   rows=dictionary.get('rows'),
                   columns=dictionary.get('columns'),
                   embeded_text=dictionary.get('embeded_text'))


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

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'SeparatorRegion':
        return cls(**super().from_dict(dictionary))


class Border(BaseElement):
    """
    Border of the actual page (if the scanned image contains parts not belonging to the page).
    """

    tag = 'Border'

    def __init__(self, coords: List[Point]=None, id: str = None):
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

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'Border':
        return cls(coords=Point.list_to_point(dictionary.get('coords')))

    def to_dict(self, non_serializable_keys: List[str]=list()) -> dict:
        if 'coords' in vars(self).keys() and 'coords' not in non_serializable_keys:
            non_serializable_keys += ['coords']
        return json_serialize(vars(self), non_serializable_keys=non_serializable_keys)


class Metadata(BaseElement):
    """
    Metadata of PAGE XML
    """
    tag = 'Metadata'

    def __init__(self, creator: str=None, created: str=None, last_change: str=None, comments: str=None):
        self.creator = creator
        self.created = created
        self.last_change = last_change
        self.comments = comments if comments is not None else ''

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'Metadata':
        if e is None:
            return None
        cls.check_tag(e.tag)
        creator_et = e.find('p:Creator', _ns)
        created_et = e.find('p:Created', _ns)
        last_change_et = e.find('p:LastChange', _ns)
        comments_et = e.find('p:Comments', _ns)
        return Metadata(creator=creator_et.text if creator_et is not None else None,
                        created=created_et.text if created_et is not None else None,
                        last_change=last_change_et.text if last_change_et is not None else None,
                        comments=comments_et.text if comments_et is not None else None)

    def to_xml(self) -> ET.Element:
        metadata_et = ET.Element('Metadata')
        creator_et = ET.SubElement(metadata_et, 'Creator')
        creator_et.text = self.creator if self.creator is not None else ''
        created_et = ET.SubElement(metadata_et, 'Created')
        created_et.text = self.created if self.created is not None else ''
        last_change_et = ET.SubElement(metadata_et, 'LastChange')
        last_change_et.text = self.last_change if self.last_change is not None else ''
        comments_et = ET.SubElement(metadata_et, 'Comments')
        comments_et.text = self.comments if self.comments is not None else ''

        return metadata_et

    def to_dict(self):
        return vars(self)

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'Metadata':
        return cls(created=dictionary.get('created'),
                   creator=dictionary.get('creator'),
                   last_change=dictionary.get('last_change'),
                   comments=dictionary.get('comments')
                   )


class GroupSegment(Region):
    """
    Only for JSON export (no PAGE XML correspondence).
    GroupSegment is a region containing several TextLineRegions and that form a bigger region.
    It is used mainly to make line / column regions.
    """
    def __init__(self, id: str = None, coords: List[Point] = None, segment_ids: List[str] = None):
        super().__init__(id=id, coords=coords)
        self.segment_ids = segment_ids if segment_ids is not None else []

    @classmethod
    def from_dict(cls, dictionary: dict) -> 'GroupSegment':
        return cls(**super().from_dict(dictionary))


class Page(BaseElement):
    tag = 'Page'

    def __init__(self, image_filename: str=None, image_width: int=None, image_height: int=None,
                 text_regions: List[TextRegion]=None, graphic_regions: List[GraphicRegion]=None,
                 page_border: Border=None, separator_regions: List[SeparatorRegion]=None,
                 table_regions: List[TableRegion]=None, metadata: Metadata=None,
                 line_groups: List[GroupSegment]=None, column_groups: List[GroupSegment]=None):
        self.image_filename = image_filename
        self.image_width = _try_to_int(image_width)
        self.image_height = _try_to_int(image_height)
        self.text_regions = text_regions if text_regions is not None else []
        self.graphic_regions = graphic_regions if graphic_regions is not None else []
        self.page_border = page_border if page_border is not None else []
        self.separator_regions = separator_regions if separator_regions is not None else []
        self.table_regions = table_regions if table_regions is not None else []
        self.metadata = metadata if metadata is not None else Metadata()
        self.line_groups = line_groups if line_groups is not None else []
        self.column_groups = column_groups if column_groups is not None else []

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
        return cls(image_filename=dictionary.get('image_filename'),
                   image_height=dictionary.get('image_height'),
                   image_width=dictionary.get('image_width'),
                   metadata=Metadata.from_dict(dictionary.get('metadata')),
                   text_regions=[TextRegion.from_dict(tr) for tr in dictionary.get('text_regions', list())],
                   page_border=Border.from_dict(dictionary.get('page_border', dict())),
                   separator_regions=[SeparatorRegion.from_dict(sep) for sep in dictionary.get('separator_regions', list())],
                   graphic_regions=[GraphicRegion.from_dict(gr) for gr in dictionary.get('graphic_regions', list())],
                   table_regions=[TableRegion.from_dict(tr) for tr in dictionary.get('table_regions', list())],
                   line_groups=[GroupSegment.from_dict(lr) for lr in dictionary.get('line_groups', list())],
                   column_groups=[GroupSegment.from_dict(cr) for cr in dictionary.get('column_groups', list())]
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
        if self.page_border:
            page_et.append(self.page_border.to_xml())
        for sep in self.separator_regions:
            page_et.append(sep.to_xml())
        for tr in self.table_regions:
            page_et.append(tr.to_xml())
        # if self.metadata:
        #     page_et.append(self.metadata.to_xml())
        return page_et

    def write_to_file(self, filename, creator_name='dhSegment', comments=''):

        def _write_xml():
            root = ET.Element('PcGts')
            root.set('xmlns', _ns['p'])

            root.append(self.metadata.to_xml())
            root.append(self.to_xml())
            for k, v in _attribs.items():
                root.attrib[k] = v
            ET.ElementTree(element=root).write(filename)

        def _write_json():
            self_dict = vars(self)

            # json_dict = dict()
            serializable_keys = ['image_filename', 'image_height', 'image_width']
            json_dict = json_serialize(self_dict, [k for k in self_dict.keys() if k not in serializable_keys])

            with open(filename, 'w', encoding='utf8') as file:
                json.dump(json_dict, file, indent=4, sort_keys=True, allow_nan=False)

        # Updating metadata
        self.metadata.creator = creator_name
        self.metadata.comments += comments
        generated_on = str(datetime.datetime.now().isoformat())
        if self.metadata.created is None:
            self.metadata.created = generated_on
        else:
            self.metadata.last_change = generated_on

        # Depending on the extension write xml or json file
        extension = os.path.splitext(filename)[1]

        if extension == '.xml':
            _write_xml()
        elif extension == '.json':
            _write_json()
        else:
            print('WARN : No extension for export, XML export by default')
            _write_xml()

    def draw_baselines(self, img_canvas: np.ndarray, color: Tuple[int, int, int]=(255, 0, 0), thickness: int=2,
                       endpoint_radius: int=4, autoscale: bool=True):
        """
        Given an image, draws the TextLines.baselines.

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: the thickness of the line
        :param endpoint_radius: the radius of the endpoints of line s(first and last coordinates of line)
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
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

    def draw_lines(self, img_canvas: np.ndarray, color: Tuple[int, int, int]=(255, 0, 0), thickness: int=2,
                   fill: bool=True, autoscale: bool=True):
        """
        Given an image, draws the polygons containing text lines, i.e TextLines.coords

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: the thickness of the line
        :param fill: if True fills the polygon
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
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

    def draw_text_regions(self, img_canvas: np.ndarray, color: Tuple[int, int, int]=(255, 0, 0), fill: bool=True,
                          thickness: int=3, autoscale: bool=True):
        """
        Given an image, draws the TextRegions, either fills it (fill=True) or draws the contours (fill=False)

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=True the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
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
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
        :return: img_canvas is updated inplace
        """

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        border_coords = (Point.list_to_cv2poly(self.page_border.coords) * ratio).astype(np.int32) \
            if len(self.page_border.coords) > 0 else []
        if fill:
            cv2.fillPoly(img_canvas, [border_coords], color)
        else:
            cv2.polylines(img_canvas, [border_coords], True, color, thickness=thickness)

    def draw_separator_lines(self, img_canvas: np.ndarray, color: Tuple[int, int, int]=(0, 255, 0),
                             thickness: int=3, filter_by_id: str='', autoscale: bool=True):
        """
        Given an image, draws the SeparatorRegion.

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param thickness: thickness of the line
        :param filter_by_id: string to filter the lines by id. For example vertical/horizontal lines can be filtered
                             if 'vertical' or 'horizontal' is mentioned in the id.
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
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

    def draw_graphic_regions(self, img_canvas: np.ndarray, color: Tuple[int, int, int]=(255, 0, 0),
                             fill: bool=True, thickness: int=3, autoscale: bool=True):
        """
        Given an image, draws the GraphicRegions, either fills it (fill=True) or draws the contours (fill=False)

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=True the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
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

    def draw_text(self, img_canvas, color: Tuple[int, int, int]=(255, 0, 0), thickness: int=5,
                  font=cv2.FONT_HERSHEY_SIMPLEX, font_scale: float=1.0, autoscale: bool=True):
        text_lines = [tl for tr in self.text_regions for tl in tr.text_lines]
        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        tl_tuple_coords_text = [((np.array(Point.point_to_list(tl.coords)) * ratio).astype(np.int32),
                                 tl.text.text_equiv) for tl in text_lines if len(tl.coords) > 0]

        for coords, text in tl_tuple_coords_text:
            polyline = Polygon(coords)
            xmin, ymin, xmax, ymax = polyline.bounds
            ymin = np.maximum(0, ymin - 20)

            cv2.putText(img_canvas, text, (int(xmin), int(ymin)), fontFace=font, fontScale=font_scale, color=color,
                        thickness=thickness)

    def draw_line_groups(self, img_canvas: np.array, color: Tuple[int, int, int]=(0, 255, 0), fill: bool=False,
                         thickness: int=5,  autoscale: bool=True):
        """
        This is only valid when parsing JSON files. It will draw line groups

        :param img_canvas: 3 channel image in which the region will be drawn
        :param color: (R, G, B) value color
        :param fill: either to fill the region (True) of only draw the external contours (False)
        :param thickness: in case fill=False the thickness of the line
        :param autoscale: whether to scale the coordinates to the size of img_canvas. If True,
                          it will use the dimensions provided in Page.image_width and Page.image_height
                          to compute the scaling ratio
        :return:
        """
        assert self.line_groups, "There is no Line group"

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        lg_coords = [(Point.list_to_cv2poly(lg.coords)*ratio).astype(np.int32) for lg in self.line_groups
                     if len(lg.coords) > 0]
        if fill:
            cv2.fillPoly(img_canvas, lg_coords, color)
        else:
            cv2.polylines(img_canvas, lg_coords, True, color, thickness=thickness)

    def draw_column_groups(self, img_canvas: np.array, color: Tuple[int, int, int]=(0, 255, 0), fill: bool=False,
                           thickness: int=5,  autoscale: bool=True):

        assert self.column_groups, "There is no Line group"

        if autoscale:
            assert self.image_height is not None
            assert self.image_width is not None
            ratio = (img_canvas.shape[0]/self.image_height, img_canvas.shape[1]/self.image_width)
        else:
            ratio = (1, 1)

        cg_coords = [(Point.list_to_cv2poly(cg.coords)*ratio).astype(np.int32) for cg in self.column_groups
                     if len(cg.coords) > 0]
        if fill:
            cv2.fillPoly(img_canvas, cg_coords, color)
        else:
            cv2.polylines(img_canvas, cg_coords, True, color, thickness=thickness)


def parse_file(filename: str) -> Page:
    """
    Parses the files to create the corresponding Page object. The files can be a .xml or a .json.

    :param filename: file to parse (either json of page xml)
    :return: Page object containing all the parsed elements
    """
    extension = os.path.splitext(filename)[1]

    if extension == '.xml':
        xml_page = ET.parse(filename)
        page_elements = xml_page.find('p:Page', _ns)
        metadata_et = xml_page.find('p:Metadata', _ns)
        page = Page.from_xml(page_elements)
        page.metadata = Metadata.from_xml(metadata_et)
        return page
    elif extension == '.json':
        with open(filename, 'r', encoding='utf8') as file:
            json_dict = json.load(file)
        return Page.from_dict(json_dict)
    else:
        raise NotImplementedError


def json_serialize(dict_to_serialize: dict, non_serializable_keys: List[str]=list()) -> dict:

    new_dict = dict_to_serialize.copy()
    for key in non_serializable_keys:
        if isinstance(new_dict[key], list):
            new_dict[key] = [elem.to_dict() for elem in new_dict[key]] if new_dict[key] else []
        elif isinstance(new_dict[key], np.ndarray):
            new_dict[key] = new_dict[key].tolist()
        else:
            new_dict[key] = new_dict[key].to_dict()

    return new_dict


def save_baselines(filename, baselines, ratio=(1, 1), initial_shape=None):
    txt_lines = [TextLine.from_array(baseline_coords=b, id='line_{}'.format(i)) for i, b in enumerate(baselines)]
    for l in txt_lines:
        l.scale_baseline_points(ratio)
    txt_region = TextRegion(text_lines=txt_lines, id='region_0')
    page = Page(text_regions=[txt_region],
                image_height=int(initial_shape[0]*ratio[0]) if initial_shape is not None else None,
                image_width=int(initial_shape[1]*ratio[1]) if initial_shape is not None else None)
    page.write_to_file(filename)
