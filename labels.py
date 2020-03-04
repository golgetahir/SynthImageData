import xml.etree.ElementTree as ET


def parse_label_file(file):
    all_insts = []

    try:
        tree = ET.parse(file)
    except Exception as e:
        print(e)
        print('Ignore this bad annotation: ' + file)

    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))

            all_insts += [obj]
    return all_insts

def generate_one_xml(path, synt_fod,frame_id):

    fod_id = str(synt_fod.fod.frame_id)
    filename = str(frame_id)
    image = synt_fod.image
    fod_type = synt_fod.fod.type
    rot_angle = str(synt_fod.fod.rotation)
    scaling = str(synt_fod.fod.scaling)
    bbox = synt_fod.fod.bbox

    height, width, channels = image.shape
    with open(path + "\\" + filename + ".xml", "w") as f:
        f.write('<annotation verified="yes">                             \n')
        f.write('    <filename>' + 'nb_' + filename + '.png</filename>   \n')
        f.write('    <size>                                              \n')
        f.write('        <width>' + str(width) + '</width>               \n')
        f.write('        <height>' + str(height) + '</height>            \n')
        f.write('        <depth> ' + str(channels) + ' </depth>          \n')
        f.write('    </size>                                             \n')
        f.write('    <FOD>                                               \n')
        f.write('        <ID>' + fod_id + '</ID>                         \n')
        f.write('        <type>' + fod_type + '</type>                   \n')
        f.write('        <rotation>' + rot_angle + '</rotation>          \n')
        f.write('        <scaling>' + scaling + '</scaling>              \n')
        f.write('        <bndbox>                                        \n')
        f.write('            <xmin>' + str(bbox[0]) + '</xmin>           \n')
        f.write('            <ymin>' + str(bbox[2]) + '</ymin>           \n')
        f.write('            <xmax>' + str(bbox[1]) + '</xmax>           \n')
        f.write('            <ymax>' + str(bbox[3]) + '</ymax>           \n')
        f.write('        </bndbox>                                       \n')
        f.write('    </FOD>                                              \n')

        f.write('</annotation>                                            ')
