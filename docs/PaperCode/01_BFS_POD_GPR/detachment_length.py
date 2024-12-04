import dolfinx
import numpy as np
from dolfinx import geometry
from pyforce.tools.functions_list import FunctionsList
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def extract_cells(domain: dolfinx.mesh.Mesh, points: np.ndarray):
    """
    This function can be used to extract data along a line defined by the variables `points`, crossing the domain.
 
    Parameters
    ----------
    domain  : dolfinx.mesh.Mesh
        Domain to extract data from.
    points : np.ndarray 
        Points listing the line from which data are extracted.

    Returns
    -------
    xPlot : np.ndarray 
        Coordinate denoting the cell from which data are extracted.
    cells : list
        List of cells of the mesh.
    """
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    xPlot = np.array(points_on_proc, dtype=np.float64)

    return xPlot, cells


class detach_x_bfs():
    def __init__(self,  domain: dolfinx.mesh.Mesh,
                        Nhplot = 125, x_bound = [0, 0.14],
                        y_line = -0.0045):


        self.Nhplot = Nhplot
        x_line = np.linspace(x_bound[0], x_bound[1] + 1e-20, self.Nhplot)
        
        self.points = np.zeros((3, self.Nhplot))
        self.points[0] = x_line
        self.points[1] = y_line
        self._xPlot, self._cells = extract_cells(domain, self.points)
    
    def compute(self, snaps: FunctionsList):
        ux_line = np.zeros((self.Nhplot, len(snaps)))
        _detach = np.zeros((len(snaps), ))
        
        for tt in range(len(snaps)):
            ux_line[:, tt] = snaps.map(tt).eval(self._xPlot, self._cells)[:, 0]
            
            sign_changes = np.where(np.diff(np.sign(ux_line[:, tt])))[0]
            sign_change_points = self._xPlot[:,0][sign_changes]
            
            if len(sign_change_points) > 0:
                _interp = interp1d(self._xPlot[:,0], ux_line[:, tt], fill_value='extrapolate')
                roots = fsolve(_interp, x0=sign_change_points)
            
                _detach[tt] = max(roots)
            else: 
                _detach[tt] 
                
        return _detach