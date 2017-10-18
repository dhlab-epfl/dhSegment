from xml.etree import ElementTree as ET
from typing import List, Optional, Union

_ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}


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
        t = e.attrib['points']
        result = []
        for p in t.split(' '):
            values = p.split(',')
            assert len(values) == 2
            x, y = int(values[0]), int(values[1])
            result.append(Point(y,x))
        return result


class BaseElement:
    tag = None

    @classmethod
    def full_tag(cls):
        return '{{{}}}{}'.format(_ns['p'], cls.tag)

    @classmethod
    def check_tag(cls, tag):
        assert tag == cls.full_tag(), 'Invalid tag, expected {} got {}'.format(cls.full_tag(), tag)


class TextLine(BaseElement):
    tag = 'TextLine'

    def __init__(self, id=None, coords=None, baseline=None, text_equiv=''):
        self.coords = coords if coords is not None else []  # type: List[Point]
        self.baseline = baseline if baseline is not None else []  # type: List[Point]
        self.id = id  # type: Optional[str]
        self.text_equiv = text_equiv  # type: str

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextLine':
        cls.check_tag(e.tag)
        return TextLine(
            id=e.attrib.get('id'),
            coords=Point.list_from_xml(e.find('p:Coords', _ns)),
            baseline=Point.list_from_xml(e.find('p:Baseline', _ns)),
            text_equiv=_get_text_equiv(e)
        )


class TextRegion(BaseElement):
    tag = 'TextRegion'

    def __init__(self, id=None, coords=None, text_lines=None, text_equiv=''):
        self.id = id  # type: Optional[str]
        self.coords = coords if coords is not None else []  # type: List[Point]
        self.text_equiv = text_equiv  # type: str
        self.text_lines = text_lines if text_lines is not None else []  # type: List[TextLine]

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'TextRegion':
        cls.check_tag(e.tag)
        return TextRegion(
            id=e.attrib.get('id'),
            coords=Point.list_from_xml(e.find('p:Coords', _ns)),
            text_lines=[TextLine.from_xml(tl) for tl in e.findall('p:TextLine', _ns)],
            text_equiv=_get_text_equiv(e)
        )


class Page(BaseElement):
    tag = 'Page'

    def __init__(self, image_filename=None, image_width=None, image_height=None, text_regions=None):
        self.image_filename = image_filename  # type: Optional[str]
        self.image_width = _try_to_int(image_width)  # type: Optional[int]
        self.image_height = _try_to_int(image_height)  # type: Optional[int]
        self.text_regions = text_regions if text_regions is not None else []  # type: List[TextRegion]

    @classmethod
    def from_xml(cls, e: ET.Element) -> 'Page':
        cls.check_tag(e.tag)
        return Page(
            image_filename=e.attrib.get('imageFilename'),
            image_width=e.attrib.get('imageWidth'),
            image_height=e.attrib.get('imageHeight'),
            text_regions=[TextRegion.from_xml(tr) for tr in e.findall('p:TextRegion', _ns)]
        )


def parse_file(filename: str) -> Page:
    xml_page = ET.parse(filename)
    page_elements = xml_page.findall('p:Page', _ns)
    #TODO can there be multiple pages in a single XML file?
    assert len(page_elements) == 1
    return Page.from_xml(page_elements[0])
