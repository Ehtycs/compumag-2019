import deps

import geometry_tools as gt
from itertools import chain

import numpy as np

# density around the conductors
# must be somewhat smaller than conductor radius
#density_windings = 0.0005
densities = {'density_windings': 0.0005,
             # density in the coupling boundary in sub domain
             'density_coupling_sub': 0.003,
             # density in coupling boundary in main domain
             'density_coupling_main': 0.005,
             # density at "infinity"
             'density_inf': 0.03 }

def make_density_config(**kwargs):
    
    #d = {k: kwargs.pop(k, densities[k]) for k, v in densities.items()}

    #assert not kwargs, f"Unexpected arguments in kwargs {list(kwargs.keys())}"
    
    def fun(field):
        return kwargs[field]
    
    return fun

class sub:
    """ Sub domain definitions. Inside this class just for the namespace
    effect.
    """
    
    # sub domain dimensions
    width = 0.106
    height = 0.01
    center = (width/2, height/2)
    
    nturns = 6
    crad = 0.001
    
    AIR = (2,1)
    WIND_LEFT_FST = 100
    WIND_RIGHT_FST = 200
    BDRY = (1,4)
    
    @classmethod
    def embedding(cls, cp_pos, ang):
        # location of bottom left corner in global coordinates
        xcp, ycp = cp_pos
        xc = xcp - cls.width/2
        yc = ycp - cls.height/2
        
        rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                        [np.sin(ang),  np.cos(ang), 0],
                        [0          ,            0, 1]])
        eye = np.eye(3)
        rot_trans = np.array([[0,0,xcp],[0,0,ycp],[0,0,0]])
        
        trans = np.array([[0,0,xc],[0,0,yc],[0,0,0]])
        
        def fun(x_in):
            
            if(x_in.ndim < 2):
                xs = x_in[None, 0:2]
            xs = x_in[:, 0:2]
            
            x = np.concatenate([xs, np.ones((x_in.shape[0],1))], axis=1)
                      
            oper = (eye+rot_trans) @ rot @ (eye-rot_trans) @ (eye+trans)

            newx = (oper[None,...] @ x[...,None]).squeeze(2)
            newx[:,2] = x_in[:,2]
            return newx
        
        return fun

    @classmethod
    def create(cls, gmsh, densities={}):
        model = gmsh.model
        factory = model.occ
        
        dens_conf = make_density_config(**densities)

        density_windings = dens_conf('density_windings')
        # density in the coupling boundary in sub domain
        density_coupling_sub = dens_conf('density_coupling_sub')


    
        x1 = 0
        x2 = cls.width
        y1 = 0
        y2 = cls.height
        
        dx = x2-x1
        dy = y2-y1
        
        rec = factory.addRectangle(x1, y1, 0, dx, dy)  
        factory.synchronize()
        bdry = model.getBoundary([(2,rec)])
        
        ltags = gt.add_half_spiral_2d(factory, 
                                      (0.003, 0.005, 0),
                                      (0.038, 0.005, 0), 
                                      cls.nturns, cls.crad)
        
        rtags = gt.add_half_spiral_2d(factory, 
                                      (0.068, 0.005, 0),
                                      (0.103, 0.005, 0), 
                                      cls.nturns, cls.crad)
        
        ldimtags = [(2,t) for t in ltags]
        rdimtags = [(2,t) for t in rtags]
        
        tmp = factory.fragment([(2, rec)], ldimtags+rdimtags)
        
        lconds = [v[1] for v in tmp[0][0:cls.nturns]]
        rconds = [v[1] for v in tmp[0][cls.nturns:-1]]
        rec = tmp[0][-1][1]
                
        factory.synchronize()
        
        gmsh.model.addPhysicalGroup(2, [rec], cls.AIR[1])
        gmsh.model.setPhysicalName(2, cls.AIR[1], "Air")
        
        # add physical tags for left conductors
        left_tag_range = range(cls.WIND_LEFT_FST, 
                               cls.WIND_LEFT_FST+len(lconds))
        
        for phystag, cond in zip(left_tag_range, lconds):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)

        # add physical tags for right conductors
        right_tag_range = range(cls.WIND_RIGHT_FST, 
                                cls.WIND_RIGHT_FST+len(rconds))
        
        for phystag, cond in zip(right_tag_range, rconds):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)
    
        boundary = [t for _,t in bdry]
        gmsh.model.addPhysicalGroup(1, boundary, cls.BDRY[1])
        gmsh.model.setPhysicalName(1, cls.BDRY[1], "Boundary")
        
        bpoints = gmsh.model.getBoundary((2,rec), True, True, True)

        gmsh.model.mesh.setSize(bpoints[0:4], density_coupling_sub)
        gmsh.model.mesh.setSize(bpoints[4:], density_windings)
        
#        model.mesh.setSize([(0,1), (0,2), (0,3), (0,4)], density_coupling_sub)
#        model.mesh.setSize([(0,5), (0,6)], density_windings)

class main:
    """ Main domain definitions. Inside this class just for the namespace
    effect.
    
    This should take into account the holes needed to be made for subdomains
    """
    
    
    # For simulations    
    bl_corner = (-0.250, -0.100, 0)
    dimensions = (0.500, 0.200)
    
    # For pretty plots of the mesh
#    dimensions = (0.300, 0.150)
#    bl_corner = (-dimensions[0]/2, -dimensions[1]/2, 0)
    
    AIR = (2,1)
    SLOTU = (1,2)
    SLOTD = (1,3)
    BDRY = (1,4)
    
    @classmethod
    def create(cls, gmsh, holes, tilts=None, densities={}):
        """ hole1 and hole2 give the coordinates of the centers of the holes 
        """
        factory = gmsh.model.occ
        
        dens_conf = make_density_config(**densities)
        # density in coupling boundary in main domain
        density_coupling_main = dens_conf('density_coupling_main')
        # density at "infinity"
        density_inf = dens_conf('density_inf')
    
        # main domain area
        dom = factory.addRectangle(*cls.bl_corner, *cls.dimensions)
        
        dx = sub.width
        dy = sub.height
        
        hole_dimtags = [(2, factory.addRectangle(x-dx/2, y-dy/2, 0, dx, dy))
                        for x, y in holes]
        
        if tilts is not None:
            factory.synchronize()

            for dt,b in zip(hole_dimtags, tilts):
                x,y,angle = b
                factory.rotate([dt], x, y, 0, 0, 0, 1, angle)
        
        out = factory.cut([(2, dom)], hole_dimtags)
        dom = out[0][0][1]
        
        factory.synchronize()
        
        bdry = [t for _, t in gmsh.model.getBoundary([(2,dom)])]

        gmsh.model.addPhysicalGroup(2, [dom], cls.AIR[1])
        gmsh.model.setPhysicalName(2, cls.AIR[1], "Air")
        gmsh.model.addPhysicalGroup(1, bdry[0:4], cls.SLOTU[1])
        gmsh.model.setPhysicalName(1, cls.SLOTU[1], "SlotU")
        gmsh.model.addPhysicalGroup(1, bdry[4:8], cls.SLOTD[1])
        gmsh.model.setPhysicalName(1, cls.SLOTD[1], "SlotD")
    
        gmsh.model.addPhysicalGroup(1, bdry[8:12], cls.BDRY[1])
        gmsh.model.setPhysicalName(1, cls.BDRY[1], "Boundary")
        
        #set mesh density
        points = gmsh.model.getBoundary([(2,1)], True, True, True)
        
        gmsh.model.mesh.setSize(points[0:8], density_coupling_main)
    
        gmsh.model.mesh.setSize(points[8:12], density_inf)

class combined:
    
    AIR = (2,1)
    AIR_COIL = (2,2) 
    UWIND_LEFT_FST = 100
    UWIND_RIGHT_FST = 200
    LWIND_LEFT_FST = 300
    LWIND_RIGHT_FST = 400
    BDRY = (1,6)
    
    @classmethod
    def create(cls, gmsh, holes, tilts=None, densities={}):
        
        dens_conf = make_density_config(**densities)

        density_windings = dens_conf('density_windings')
        # density at "infinity"
        density_inf = dens_conf('density_inf')
        
        density_coupling_sub = dens_conf('density_coupling_sub')

        
        factory = gmsh.model.occ
    
        # main domain area
        dom = factory.addRectangle(*main.bl_corner, *main.dimensions)
        
        dx = sub.width
        dy = sub.height
        
        coil_dimtags = [[],[]]
        rect_dimtags = []
        #
#        hole_dimtags = [(2, factory.addRectangle(x-dx/2, y-dy/2, 0, dx, dy))
#                        for x, y in holes]
#        
#        if tilts is not None:
#            factory.synchronize()
#
#            for dt,b in zip(hole_dimtags, tilts):
#                x,y,angle = b
#                factory.rotate([dt], x, y, 0, 0, 0, 1, angle)
#        
#        out = factory.cut([(2, dom)], hole_dimtags)
#        dom = out[0][0][1]
#        
#        factory.synchronize()
        #
                
        for nhole, coords in enumerate(holes):
            xh, yh = coords
            # bottom left corner
            x = xh-0.5*dx
            y = yh-0.5*dy
            
            ltags = gt.add_half_spiral_2d(factory, 
                                          (x+0.003, y+0.005, 0),
                                          (x+0.038, y+0.005, 0 ), 
                                          sub.nturns, sub.crad)
            
            rtags = gt.add_half_spiral_2d(factory, 
                                          (x+0.068, y+0.005, 0),
                                          (x+0.103, y+0.005, 0 ), 
                                          sub.nturns, sub.crad)
        
            # add a box around the winding to enforce mesh density
            rect = factory.addRectangle(x, y, 0, sub.width, sub.height)
        
            ldimtags = [(2,t) for t in ltags]
            rdimtags = [(2,t) for t in rtags]

            if tilts is not None:
                factory.synchronize()
                xr, yr, ang = tilts[nhole]
                factory.rotate(ldimtags+rdimtags, xr, yr, 0, 0, 0, 1, ang)
                factory.rotate([(2, rect)], xr, yr, 0, 0, 0, 1, ang)
        
            coil_dimtags[nhole] = ldimtags+rdimtags
            rect_dimtags.append((2,rect))
        
        dimtags = list(chain.from_iterable(coil_dimtags))

        # first fragment conductors and the rectangles
        # then fragment the rectangles and the air
        # fragment returns a tuple of lists of dimtags...
        tmp3 = factory.fragment([(2, dom)], rect_dimtags)
        
        rec = tmp3[0][2][1]
        recu = tmp3[0][0][1]
        recl = tmp3[0][1][1]
        
        tmp1 = factory.fragment(rect_dimtags[0:1], coil_dimtags[0])        
        tmp2 = factory.fragment(rect_dimtags[1:], coil_dimtags[1])
        
        nt = sub.nturns
        
        recu = tmp1[0][-1][1]
        lcondsu = [v[1] for v in tmp1[0][0:nt]]
        rcondsu = [v[1] for v in tmp1[0][nt:2*nt]]

        recl = tmp2[0][-1][1]        
        lcondsl = [v[1] for v in tmp2[0][0:nt]]
        rcondsl = [v[1] for v in tmp2[0][nt:2*nt]]
        
        factory.synchronize()

        # add physical tags for left conductors
        uleft_tag_range = range(cls.UWIND_LEFT_FST, 
                               cls.UWIND_LEFT_FST+len(lcondsu))
        
        for phystag, cond in zip(uleft_tag_range, lcondsu):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)
        
        # add physical tags for right conductors
        uright_tag_range = range(cls.UWIND_RIGHT_FST, 
                                cls.UWIND_RIGHT_FST+len(rcondsu))
        
        for phystag, cond in zip(uright_tag_range, rcondsu):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)
            
        # add physical tags for left conductors
        lleft_tag_range = range(cls.LWIND_LEFT_FST, 
                               cls.LWIND_LEFT_FST+len(lcondsl))
        
        for phystag, cond in zip(lleft_tag_range, lcondsl):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)

        # add physical tags for right conductors
        lright_tag_range = range(cls.LWIND_RIGHT_FST, 
                                cls.LWIND_RIGHT_FST+len(rcondsl))
        
        for phystag, cond in zip(lright_tag_range, rcondsl):            
            gmsh.model.addPhysicalGroup(2, [cond], phystag)
                    
        bdry = [t for _, t in gmsh.model.getBoundary([(2,rec)])]
        
        gmsh.model.addPhysicalGroup(2, [rec], cls.AIR[1])
        gmsh.model.setPhysicalName(2, cls.AIR[1], "Air")
        gmsh.model.addPhysicalGroup(2, [recl, recu], cls.AIR_COIL[1])
        gmsh.model.setPhysicalName(2, cls.AIR_COIL[1], "AirCoils")
        gmsh.model.addPhysicalGroup(1, bdry[0:4], cls.BDRY[1])
        gmsh.model.setPhysicalName(1, cls.BDRY[1], "Boundary")

        factory.synchronize()
        
        #set mesh density on air and cpl boundary
        points = gmsh.model.getBoundary((2,rec), True, True, True)
        
        gmsh.model.mesh.setSize(points[4:], density_coupling_sub)        
        gmsh.model.mesh.setSize(points[0:4], density_inf)
        
        #set mesh dens on cpl boundary and windings, copy & paste abstraction!
        points = gmsh.model.getBoundary((2,recu), True, True, True)
        
        gmsh.model.mesh.setSize(points[4:], density_windings)        
        gmsh.model.mesh.setSize(points[0:4], density_coupling_sub)

        points = gmsh.model.getBoundary((2,recl), True, True, True)
        
        gmsh.model.mesh.setSize(points[4:], density_windings)        
        gmsh.model.mesh.setSize(points[0:4], density_coupling_sub)
