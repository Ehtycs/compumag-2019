blocknames = {
    'MeshFormat': 'MeshFormat',
    'Nodes': 'Nodes',
    'Elements': 'Elements',
    'ElementTags': 'ElementTags',
    'PhysicalNames': 'PhysicalNames',
    'NodeData': 'NodeData',
    'ElementData':'ElementData'
}
elm_names = {
    1:'line2',
    2:'triangle3',
    3:'quadrangle4',
    4:'tetrahedron4',
    5:'hexahedron8',
    6:'prism6',
    7:'pyramid5',
    8:'line2',
    9:'triangle6',
    10:'quadrangle9',
    11:'tetrahedron10',
    12:'hexahedron27',
    13:'prism18',
    14:'pyramid14',
    15:'point1',
    16:'quadrangle8'
}
elm_numbers = {
    'line2':1,
    'triangle3':2,
    'quadrangle4':3,
    'tetrahedron4':4,
    'hexahedron8':5,
    'prism6':6,
    'pyramid5':7,
    'line3':8,
    'triangle6':9,
    'quadrangle9':10,
    'tetrahedron10':11,
    'hexahedron27':12,
    'prism18':13,
    'pyramid14':14,
    'point1':15,
    'quadrangle8':16
}


#compatibility with meshio
def mio_parse(filename):
    meshio_naming = {
        'line2':'line',
        'triangle3':'triangle',
        'quadrangle4':'quad',
        'tetrahedron4':'tetra',
        'hexahedron8':'hexahedron',
        'prism6':'wedge',
        'pyramid5':'pyramid',
        'line3':'line3',
        'triangle6':'triangle6',
        'quadrangle9':'quad9',
        'tetrahedron10':'tetra10',
        'hexahedron27':'hexahedron27',
        'prism18':'prism18',
        'pyramid14':'pyramid14',
        'point1':'vertex',
        'quadrangle8':'quad8'
    }
    import numpy as np
    msh_dict = parse(filename)
    nodes =  np.array(msh_dict[blocknames['Nodes']])
    elements = msh_dict[blocknames['Elements']]
    
    elements = { meshio_naming[outer_k]: np.array(outer_v)-1
                    for outer_k, outer_v in elements.items()
                }

    pointdata = {}
    celldata = msh_dict[ blocknames['ElementTags'] ]

    celldata = { meshio_naming[outer_k]: {
                    inner_k: inner_v
                    for inner_k, inner_v in outer_v.items() 
                        if inner_k is 'geometrical' or inner_k is 'physical'
                }
                for outer_k, outer_v in celldata.items()
            }
    
    fielddata = msh_dict[ blocknames['PhysicalNames'] ]
    fielddata = dict( [ (fielddata[key],key) for key in fielddata.keys() ] )
    return (nodes,elements,pointdata,celldata,fielddata)


#this is the actual stuff
def parse(filename):
    with open(filename, 'r') as mshfile:
        msh_dict = _read_raw_dict(mshfile)

    msh_dict = _parse_meshformat(msh_dict)
    msh_dict = _parse_physnames(msh_dict)
    msh_dict = _parse_nodes(msh_dict)
    msh_dict = _parse_elements(msh_dict)

    return msh_dict

def _read_raw_dict(mshfile):
    raw_dict = {}
    separator = '$'
    end = 'End'
    lines = mshfile.readlines()
    blockranges = []
    blockbegin = -1
    #let us find the row-indices of the block delimiters $iiii and $Endiii
    for i in range(len(lines)):
        line = lines[i]
        if lines[i].find(separator+end) > -1:
            blockranges.append([blockbegin, i])
        elif lines[i].find(separator) > -1:
            blockbegin = i
    #next we extract the necessary lines in each dict cell
    #print(blockranges)
    for i in blockranges:
        name = lines[i[0]]
        name = name[ name.find(separator)+1:-1 ]
        raw_dict[name] = lines[i[0]+1:i[1]]

    return raw_dict

def _parse_meshformat(msh_dict):
    mf_n = blocknames['MeshFormat']
    meshformat = msh_dict[mf_n]
    meshformat = meshformat[0][:-1].split(" ")
    msh_dict[mf_n] = meshformat
    return msh_dict

def _parse_physnames(msh_dict):
    ph_n = blocknames['PhysicalNames']
    if ph_n not in msh_dict.keys():
        return msh_dict
    physnames = msh_dict[ph_n]
    physnames = [ i[:-1].split(" ")[1:] for i in physnames[1:] ]
    for i in range(len(physnames)):
        physnames[i] = [int(physnames[i][0]), physnames[i][1][1:-1]  ]
    msh_dict[ph_n] = dict(physnames)
    return msh_dict

def _parse_nodes(msh_dict):
    node_d = blocknames['Nodes']
    nodes = msh_dict[node_d]
    nodes = nodes[1:]
    nodes = [ list(map(float,i[:-1].split(" ")[1:])) for i in nodes]
    msh_dict[node_d] = nodes
    return msh_dict

def _parse_elements(msh_dict):
    element_n = blocknames['Elements']
    el_tag_n = blocknames['ElementTags']
    elements = msh_dict[element_n][1:]
    for i in range(len(elements)):
        elements[i] = elements[i][:-1].split(" ")
    element_dict, tag_dict = _parse_elements_types(elements)
    msh_dict[element_n] = element_dict
    msh_dict[el_tag_n] = tag_dict
    return msh_dict

def _parse_elements_types(elements):
    element_dict = {}
    tag_dict = {}
    phys = 'physical'
    geom = 'geometrical'
    other = 'other'
    for i in elements:
        #print (i)
        el_type = elm_names[int(i[1])]
        num_tags = int(i[2])
        if el_type not in element_dict.keys():
            element_dict[el_type] = [list(map(int,i[3+num_tags:]))]
            tag_dict[el_type] = {geom:[], phys:[], other:[]} #[i[3:3+num_tags]]
            tag_dict[el_type][phys] = [int(i[3])]
            tag_dict[el_type][geom] = [int(i[4])]
            tag_dict[el_type][other] = [list(map(int,i[5:5+num_tags-2]))]
        else:
            element_dict[el_type].append(list(map(int,i[3+num_tags:])))
            #print(tag_dict[el_type])
            tag_dict[el_type][phys].append(int(i[3]))
            tag_dict[el_type][geom].append(int(i[4]))
            tag_dict[el_type][other].append(list(map(int,i[5:5+num_tags-2])))
    return (element_dict, tag_dict)
    
def write(filename, msh_dict):
    msh_str = ""
    msh_str = msh_str + _write_mshformat(msh_dict)
    msh_str = msh_str + _write_physicalnames(msh_dict)
    msh_str = msh_str + _write_nodes(msh_dict)
    msh_str = msh_str + _write_elements(msh_dict)
    msh_str = msh_str + _write_nodedata(msh_dict)
    msh_str = msh_str + _write_elementdata(msh_dict)
    
    with open(filename, 'w') as mshfile:
        mshfile.write(msh_str)
    
def _write_mshformat(msh_dict):
    mf_n = blocknames['MeshFormat']
    mf_str = ""
    if mf_n not in msh_dict.keys():
        return mf_str
    mf_str = "$"+mf_n+"\n"
    mf_str = mf_str + " ".join( str(i) for i in msh_dict[mf_n] ) + "\n"
    mf_str = mf_str + "$End" + mf_n + "\n"
    return mf_str

def _write_physicalnames(msh_dict):
    ph_n = blocknames['PhysicalNames']
    ph_str = ""
    if ph_n not in msh_dict.keys():
        return ph_str
    ph_str = "$"+ph_n+"\n"
    ph_values = list(msh_dict[ph_n].values())
    ph_keys = list(msh_dict[ph_n].keys())
    ph_data = []
    for i in range(len(ph_keys)):
        ph_data.append( [i+1, ph_keys[i], '"'+ph_values[i]+'"'] )
    
    ph_str = ph_str + str(len(ph_data)) + "\n"
    ph_str = ph_str + "\n".join( " ".join( str(pp) for pp in i ) for i in ph_data ) + "\n"
    ph_str = ph_str + "$End" + ph_n + "\n"
    return ph_str

def _write_nodes(msh_dict):
    nod_n = blocknames['Nodes']
    nod_str = ""
    if nod_n not in msh_dict.keys():
        return nod_str
    nodedata = msh_dict[nod_n]
    nod_str = "$"+nod_n+"\n"
    nodenum = len(nodedata)
    nod_str = nod_str + str(nodenum) + "\n"
    for i in range(nodenum):
        nod_str = nod_str + str(i+1) + " " + " ".join(str(p) for p in nodedata[i]) + "\n"
    nod_str = nod_str + "$End" + nod_n + "\n"
    return nod_str

def _write_elements(msh_dict):
    #element_numbering = {}
    el_n = blocknames['Elements']
    eltag_n = blocknames['ElementTags']
    el_str = ""
    if el_n not in msh_dict.keys():
        return el_str
    elements = msh_dict[el_n]
    element_numbering = {}
    i = 1
    for els in elements:
        elems = msh_dict[el_n][els]
        tags = msh_dict[eltag_n][els]
        elementnumbers = list(range(i,i+len(elems)))
        element_numbering[els]=elementnumbers
        i = i+len(elems)
        combined = list( map( list, zip(tags['physical'],tags['geometrical']) ) )
        combined = [ combined[i] + tags['other'][i] for i in range(len(combined)) ]
        elementtypes = [ elm_numbers[els] for i in range(len(combined)) ]
        elementarray = [ [elementnumbers[i]]+[elementtypes[i]]+[len(combined[i])]+combined[i]+elems[i]
                         for i in range(len(elems))]
        #print(elementarray)
        el_str = el_str + "\n".join( " ".join( str(eae) for eae in ear) for ear in elementarray ) + "\n"
    el_str = "$"+el_n+"\n"+str(i-1)+"\n" + el_str + "$End"+el_n+"\n"
    msh_dict['#elementnumbering']=element_numbering
    return el_str

def _write_nodedata(msh_dict):
    ndd_n = blocknames['NodeData']
    ndd_str = ""
    if ndd_n not in msh_dict.keys():
        return ndd_str
    nodedata = msh_dict[ndd_n]
    for datakey in nodedata.keys():
        data = nodedata[datakey]
        ndd_str_loc = "$"+ndd_n+"\n"
        ndd_str_loc = ndd_str_loc + "1\n"
        ndd_str_loc = ndd_str_loc + "\"" + datakey[0] + "\"" + "\n"
        ndd_str_loc = ndd_str_loc + "1\n"
        ndd_str_loc = ndd_str_loc + str(datakey[1]) + "\n"
        ndd_str_loc = ndd_str_loc + "3\n"
        ndd_str_loc = ndd_str_loc + str(datakey[2]) + "\n"
        ndd_str_loc = ndd_str_loc + str(len(data[0])) + "\n"
        ndd_str_loc = ndd_str_loc + str(len(data)) + "\n"
        nodenumbering = list(range(1,len(data)+1))
        data = [ [nodenumbering[i]]+data[i] for i in range(len(data))  ]
        data = "\n".join( " ".join( str(el) for el in row) for row in data )
        ndd_str_loc = ndd_str_loc + data + "\n"
        ndd_str_loc = ndd_str_loc + "$End" + ndd_n + "\n"
        ndd_str = ndd_str + ndd_str_loc

    return ndd_str

def _write_elementdata(msh_dict):
    eld_n = blocknames['ElementData']
    eld_str = ""
    if eld_n not in msh_dict.keys():
        return eld_str
    elementdata = msh_dict[eld_n]
    
    elnum = msh_dict['#elementnumbering']

    for datakey in elementdata.keys():
        for eltype in elementdata[datakey].keys():
            data = elementdata[datakey][eltype]
            eld_str_loc = "$"+eld_n+"\n"
            eld_str_loc = eld_str_loc + "1\n"
            eld_str_loc = eld_str_loc + "\"" + datakey[0] + "\"" + "\n"
            eld_str_loc = eld_str_loc + "1\n"
            eld_str_loc = eld_str_loc + str(datakey[1]) + "\n"
            eld_str_loc = eld_str_loc + "3\n"
            eld_str_loc = eld_str_loc + str(datakey[2]) + "\n"
            datasizes = [ len(dp) for dp in data ]
            datasizes = max(datasizes)
            eld_str_loc = eld_str_loc + str(datasizes) + "\n"
            num_elements = sum( [ 1 for i in data if len(i)>0])
            eld_str_loc = eld_str_loc + str(num_elements) + "\n"
            elnumbering = elnum[eltype]
            #print(len(nodenumbering))
            #print(len(data))
            data = [ [elnumbering[pp]]+data[pp] for pp in range(len(data)) if len(data[pp])>0 ]
            data = "\n".join( " ".join( str(el) for el in row) for row in data )
            eld_str_loc = eld_str_loc + data + "\n"
            eld_str_loc = eld_str_loc + "$End" + eld_n + "\n"
            eld_str = eld_str + eld_str_loc

    return eld_str


