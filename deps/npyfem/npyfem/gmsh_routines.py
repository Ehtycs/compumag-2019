
""" Contains functions which can be applied to npstructs when 
GMSH backend is used.
"""
    
def gmsh_visualize(gmsh, view, npstruct, datatype="ElementData"):
    """ Visualize the field represented by this NPStruct in gmsh.

    Field must have dimensions
    field[3:] == (n,1) or (1,k) or (k,) or (n,) or (1,) (or (1,1))
    so rank 1 or rank 2 with either one of axis must be a singleton.
    Input:
        gmsh: a gmsh API object
        view: tag of a post-processing view in gmsh
        npstruct: the npstruct object being visualized
        datatype: "ElementData" / "GaussPointData", the string rep. 
                  of GMSH:s data format (see GMSH manual)
    """

    for eg, field in npstruct.data.items():
        etags = eg.element_tags
        ftype = field.shape[2:]
        if(len(ftype) > 2):
            raise ValueError("Visualization of tensors of type {} "
                             "not available".format(ftype))
        elif(len(ftype) == 2):
            if(ftype[0] > 1 and ftype[1] > 1):
                raise ValueError("Visualization of rank-2 tensors with "
                                 "dimensions {} not available. Only "
                                 "if one of the dims is 1 (row or column) "
                                 "vector).")
            elif(ftype[0] == 1):
                vals = field.squeeze(2)
            elif(ftype[1] == 1):
                vals = field.squeeze(3)
        else:
            vals = field
            
        ncomponents = vals.shape[-1]
        
        # vals needs to be a "vector of vectors", rank-2 array of dimension
        # (len(etags), components*intpoints) such that vals[ind,:] is a
        # flattened version of data related to element etags[ind]
        # vals[ind] = [ip1c1, ip1c2, ..., ip2c1, ip2c2,...]
        vals = vals.swapaxes(0,1).reshape((len(etags), -1))
            
        # gmsh starts element labeling from 1
        gmsh.view.addModelData(view, 0, "", datatype, etags+1,
                                    vals, 0, ncomponents)


