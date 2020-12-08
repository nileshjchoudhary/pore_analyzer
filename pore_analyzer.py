"""
________  ________  ________  _______           ________  ________   ________  ___           ___    ___ ________  _______   ________     
|\   __  \|\   __  \|\   __  \|\  ___ \         |\   __  \|\   ___  \|\   __  \|\  \         |\  \  /  /|\_____  \|\  ___ \ |\   __  \    
\ \  \|\  \ \  \|\  \ \  \|\  \ \   __/|        \ \  \|\  \ \  \\ \  \ \  \|\  \ \  \        \ \  \/  / /\|___/  /\ \   __/|\ \  \|\  \   
\ \   ____\ \  \\\  \ \   _  _\ \  \_|/__       \ \   __  \ \  \\ \  \ \   __  \ \  \        \ \    / /     /  / /\ \  \_|/_\ \   _  _\  
\ \  \___|\ \  \\\  \ \  \\  \\ \  \_|\ \       \ \  \ \  \ \  \\ \  \ \  \ \  \ \  \____    \/  /  /     /  /_/__\ \  \_|\ \ \  \\  \| 
    \ \__\    \ \_______\ \__\\ _\\ \_______\       \ \__\ \__\ \__\\ \__\ \__\ \__\ \_______\__/  / /      |\________\ \_______\ \__\\ _\ 
    \|__|     \|_______|\|__|\|__|\|_______|        \|__|\|__|\|__| \|__|\|__|\|__|\|_______|\___/ /        \|_______|\|_______|\|__|\|__|
                                                                                            \|___|/                                       
A class of functions that help you understand the pore strucutre better by calculating the distance grid, region segmentation 
by watershed algorithm, window location by touching regions, finding the local distance maxima (pocket centers), finding diffusion paths
using the Dijkstra 3D algorithm on a scalar field and visualizing any or all of these elegantly using Plotly, with full control over the
plot attributes.
"""
#* GENERAL ONE-TIME DEFINITIONS
def make_labels_h5(path_to_h5, gridname, peak_min =4.0, dist_min=0.5, apply_pbc=True, extra_maxi=[]):

    import h5py
    import numpy as np
    from skimage.morphology import extrema
    import scipy.ndimage as ndi
    from skimage.morphology import watershed

    hfile         = h5py.File(path_to_h5,'r')
    dgrid_np      = np.array(hfile[gridname])
    local_maxi    = extrema.h_maxima(dgrid_np, peak_min)
    if len(extra_maxi)>0:
        for e in extra_maxi:
            local_maxi[tuple(e)] = 1
    maxi_markers  = ndi.label(local_maxi)[0] # Identify feature sand label those in the 3D array
    masked_image  = dgrid_np>dist_min # Mask for the watershed
    # distance = ndi.distance_transform_edt(masked_image)
    # dgrid_np[dgrid_np<0]=0
    region_labels = watershed(-dgrid_np, markers=maxi_markers, mask=masked_image.astype(np.int))
    if apply_pbc:
        rlpbc         = apply_pbc2(region_labels, local_maxi)
        return rlpbc, local_maxi, dgrid_np[np.where(local_maxi)]
    else:
        return region_labels, local_maxi, dgrid_np[np.where(local_maxi)]
def make_labels_grid(dgrid_np, data, dist_min=0.5, peak_min=4.0, apply_pbc=True, extra_maxi=[]):

    import h5py
    import numpy as np
    from skimage.morphology import extrema
    import scipy.ndimage as ndi
    from skimage.morphology import watershed
    import ase 
    from ase.io import read

    # hfile         = h5py.File(path_to_h5)
    # dgrid_np      = np.array(hfile[gridname])
    
    local_maxi    = extrema.h_maxima(dgrid_np, peak_min)
    
    # data = read(path_to_cif)
    cell = ase.geometry.complete_cell(data.get_cell())
    
    # Merge local maxima that are closer than 1 A, using agglomerative clustering
    extrema_points = np.array(np.where(local_maxi)).T/local_maxi.shape # Coordinates in a 0 to 1 unit box
    extrema_points = np.dot(cell, extrema_points.T).T # True coordinates in the box
    # print('The extrema points are:')
    # print(extrema_points)
    
    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(distance_threshold=1, n_clusters=None).fit(extrema_points)
    maxima_coordinates = np.array([np.mean(extrema_points[clusters.labels_==l], axis=0) for l in np.unique(clusters.labels_)])
    
    # print('The new maxima points are: ')
    # print(maxima_coordinates)
          
    # Find the indices of the new maxima
    new_maxima_indices = (np.dot(np.linalg.inv(cell), maxima_coordinates.T).T*local_maxi.shape).astype(np.int)
    # print("New maxima indices")
    # print(new_maxima_indices)
    
    new_maxima = np.zeros(local_maxi.shape)
    new_maxima[tuple(new_maxima_indices.T)]=1
    
    if len(extra_maxi)>0:
        for e in extra_maxi:
            new_maxima[tuple(e)] = 1
    maxi_markers  = ndi.label(new_maxima)[0] # Identify feature sand label those in the 3D array
    masked_image  = dgrid_np>dist_min # Mask for the watershed
    # distance = ndi.distance_transform_edt(masked_image)
    # dgrid_np[dgrid_np<0]=0
    region_labels = watershed(-dgrid_np, markers=maxi_markers, mask=masked_image.astype(np.int))
    if apply_pbc:
        rlpbc         = apply_pbc2(region_labels, local_maxi)
        return rlpbc, local_maxi, dgrid_np[np.where(local_maxi)]
    else:
        return region_labels, local_maxi, dgrid_np[np.where(local_maxi)]
def apply_pbc3(region_labels, localmaxi):
    import numpy as np
    import copy

    def edge_mask(x):
        import numpy as np
        mask = np.ones(x.shape, dtype=bool)
        mask[x.ndim * (slice(1, -1),)] = False
        return mask


    wallmaxi = np.vstack(np.where(np.logical_and(edge_mask(region_labels), localmaxi))).T # Maimxa which fall on the wall.
    # keep_these = region_labels[np.where(np.logical_and(edge_mask(region_labels), localmaxi))] # We wanna keep these labels

    region_labels_pbc = copy.deepcopy(region_labels)

    # * Round the localmaxi which are only slightly inside the box
    rounded_wallmaxi = np.round(wallmaxi/region_labels.shape, decimals=1)

    near_walls = np.where(rounded_wallmaxi)==0 #* Roll backward for this one
    far_walls = np.where(rounded_wallmaxi)==1  #* Roll forward for this one

    #* points of the wall
    lw = rounded_wallmaxi[:,0] == 0
    rw = rounded_wallmaxi[:,0] == 1
    bw = rounded_wallmaxi[:,2] == 0
    tw = rounded_wallmaxi[:,2] == 1
    hw = rounded_wallmaxi[:,1] == 0
    fw = rounded_wallmaxi[:,1] == 1
    # print(rounded_wallmaxi)
    # print(wallmaxi[bw])
    #* Lists for each of the walls
    # lw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[region_labels.shape[0]-1, 0, 0])]] for i in range(len(wallmaxi)) if lw[i]] # left wall replace with right wall
    # rw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[-region_labels.shape[0]+1, 0, 0])]] for i in range(len(wallmaxi)) if rw[i]] # right wall replace with left wall
    # bw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[0, 0, region_labels.shape[2]-1])]] for i in range(len(wallmaxi)) if bw[i]] # bottom wall replace with top wall
    # tw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[0, 0, -region_labels.shape[2]+1])]] for i in range(len(wallmaxi)) if tw[i]] # top wall replace with bottom wall
    # hw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[0, region_labels.shape[1]-1, 0])]] for i in range(len(wallmaxi)) if hw[i]] # hind wall replace with front wall
    # fw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple(wallmaxi[i] +[0, -region_labels.shape[1]+1, 0])]] for i in range(len(wallmaxi)) if fw[i]] # front wall replace iwth hind wall


    lw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([region_labels.shape[0]-1, wallmaxi[i][1],wallmaxi[i][2]])]] for i in range(len(wallmaxi)) if lw[i]] # left wall replace with right wall
    rw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([0, wallmaxi[i][1],wallmaxi[i][2]])]] for i in range(len(wallmaxi)) if rw[i]] # right wall replace with left wall
    bw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([wallmaxi[i][0],wallmaxi[i][1], region_labels.shape[2]-1])]] for i in range(len(wallmaxi)) if bw[i]] # bottom wall replace with top wall
    tw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([wallmaxi[i][0],wallmaxi[i][1], 0])]] for i in range(len(wallmaxi)) if tw[i]] # top wall replace with bottom wall
    hw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([wallmaxi[i][0],region_labels.shape[1]-1,wallmaxi[i][2]])]] for i in range(len(wallmaxi)) if hw[i]] # hind wall replace with front wall
    fw_list = [[region_labels[tuple(wallmaxi[i])], region_labels[tuple([wallmaxi[i][0], 0, wallmaxi[i][2]])]] for i in range(len(wallmaxi)) if fw[i]] # front wall replace iwth hind wall


    merge_list = np.vstack([l for l  in [lw_list, rw_list, bw_list, tw_list, hw_list, fw_list] if len(l)!=0])
    for m in merge_list:
        if np.sum(region_labels_pbc==m[1])>0:
            region_labels_pbc[region_labels_pbc==m[1]]=m[0]



    # * Re-index the regions so that it runs from zero to N-regions.
    old_labels        = np.unique(region_labels_pbc) # This is sorted already
    number_of_regions = len(old_labels)
    new_labels        = range(number_of_regions)

    # * Replace the old-labels with new
    for i in range(number_of_regions):
        region_labels_pbc[region_labels_pbc==old_labels[i]]=new_labels[i]
    return region_labels_pbc
def clean_config(coord, cell,specie_name):
    import numpy as np
    chain_length  = {'C7':7,'C8':8,'C9':9,'C10':10,'C11':11,'C12':12,'C13':13,'C14':14,'C15':15,'C16':16}

    shifted_coord = coord - cell[0:3]*np.floor(coord/cell[0:3])
    config = shifted_coord.reshape(-1, chain_length[specie_name], 3)
    config_unshift = np.array(coord).reshape(-1, chain_length[specie_name],3)
    return shifted_coord, config, config_unshift
def get_bead_labels(region_labels, shifted_coord, cell, specie_name):
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator as RGI
    chain_length      = {'C7':7,'C8':8,'C9':9,'C10':10,'C11':11,'C12':12,'C13':13,'C14':14,'C15':15,'C16':16}
    xx                = np.linspace(0,cell[0],region_labels.shape[0])
    yy                = np.linspace(0,cell[1],region_labels.shape[1])
    zz                = np.linspace(0,cell[2],region_labels.shape[2])
    rl                = RGI((xx,yy,zz),region_labels, method='nearest')
    bead_labels       = rl(shifted_coord).reshape(-1,chain_length[specie_name])
    number_of_regions = len(np.unique(region_labels))
    molecule_groups   = []
    for i in range(number_of_regions):
        molecule_groups.append(np.unique(np.where(bead_labels==i)[0]))
    # molecule_groups.pop[0]
    return molecule_groups[1:], bead_labels
def interpolate_me(histo):
    from scipy.interpolate import RegularGridInterpolator as RGI
    import numpy as np
    xx  = np.linspace(0, 1, histo.shape[0])
    yy  = np.linspace(0, 1, histo.shape[1])
    zz  = np.linspace(0, 1, histo.shape[2])
    rgi = RGI((xx,yy,zz), histo, method='linear')
    return rgi
def find_windows_fixed(t1, rlpbc):
    from skimage.segmentation import find_boundaries
    import numpy as np
    import itertools
    from ase.io import read
    data = read(t1.file)
    A_unit = data.get_cell().T
    # import pyIsoP.grid3D as grid3D

    # path_to_cif='C:\\Users\\arung\\Google Drive\\Current Research\\ZIF-8\\pore_topo\\ZIF-8_2x2x2.cif'
    # t1         = grid3D.grid3D(path_to_cif,spacing=0.2)
    # * Define the grids
    window_centers = []
    connection_list = []
    # outer_boundaries  = find_boundaries(rlpbc, mode='outer')
    for i in itertools.combinations(range(1,np.max(rlpbc)+1),2):
        temp = np.zeros(rlpbc.shape)
        temp[rlpbc==i[0]]=i[0]
        temp[rlpbc==i[1]]=i[1]
        boundary = find_boundaries(temp.astype(np.int), mode ='outer')
        check_flag = np.logical_or(np.logical_and(temp==i[0], boundary),np.logical_and(temp==i[1], boundary))
        if np.sum(check_flag)>0:
            print("Edges found between "+str(i[0])+' and '+ str(i[1]) + ' sum: '+str(np.sum(check_flag)))
            windices = np.vstack(np.where(check_flag)).T #  * (N,3)
            window_points = np.dot(A_unit, (windices/rlpbc.shape).T).T # * (N,3)
            window_center = np.mean(window_points, axis=0)
            print(window_center)
            print('\n')
            connection_list.append([i[0], i[1]])
            window_centers.append(window_center)

    window_centers = np.round(window_centers, decimals=2)
    window_centers = np.unique(window_centers, axis=0)
    return window_centers, connection_list
def find_windows_fixed_faster(t1, rlpbc):
    from skimage.segmentation import find_boundaries
    import numpy as np
    import itertools
    from tqdm import tqdm
    from ase.io import read
    data = read(t1.file)
    A_unit = data.get_cell().T
    def check_connected(regs):
        import numpy as np
        i=regs[0]
        j=regs[1]
        # print(str(i) + ' ' + str(j))
        check_flag = np.logical_or(np.logical_and(rlpbc==i, outer[j-1]),np.logical_and(rlpbc==j, outer[i-1]))
        if np.sum(check_flag)>0:
            # print("Edges found between "+str(i)+' and '+ str(j) + ' sum: '+str(np.sum(check_flag)))
            windices = np.vstack(np.where(check_flag)).T #  * (N,3)
            window_points = np.dot(A_unit, (windices/rlpbc.shape).T).T # * (N,3)
            window_center = np.mean(window_points, axis=0)
            # print(window_center)
            # print('\n')
            return np.array([True, window_center])
        else:
            return np.array([False, None])

    # * Find the outer boundary of each individual region
    print("Step 1: FInding the outer boundaries of each region")
    outer = np.stack([find_boundaries(rlpbc==reg, mode='outer') for reg in tqdm(range(1, np.max(rlpbc)+1))])
    # print(outer.shape)
    # * Check for connectedness between regions
    print("Step 2: Checking for connectedness between regions")
    check_list = np.vstack(list(itertools.combinations(range(1,np.max(rlpbc)+1),2)))
    flag_list = np.vstack([check_connected(c) for c in tqdm(check_list)])
    print("Done!")
    return np.vstack(check_list[np.where(flag_list[:,0])]), np.round(np.vstack(flag_list[:,1][np.where(flag_list[:,0])]), decimals=2)
def beads_to_windows(config, cell, window_centers, radius):
    def dist_to_window(p, window_centers, cell, radius):
        import numpy as np
        dr = p - window_centers
        dr = dr - cell[0:3]*np.round(dr/cell[0:3]) # * Works only in orthorhombic cells
        rsq = np.sum(dr**2, axis=1)
        rsqrt = np.sqrt(rsq)
        return np.any(rsqrt<radius)
    import numpy as np
    chain_length = config.shape[1]
    coord = config.reshape(-1,3)
    flags = np.array([dist_to_window(c, window_centers, cell, radius) for c in coord])
    flags = np.vstack(flags)
    # flags.reshape(-1, chain_length, 3)
    return flags.reshape(-1, chain_length)
def plot_surface_h5(path_to_h5, gridname, t1, value, rlpbc, select='all', opacity=1.0):
    import h5py
    import numpy as np
    from ase.io import read
    data = read(t1.file)
    A_unit = data.get_cell().T

    hfile         = h5py.File(path_to_h5,'r')
    grid      = np.array(hfile[gridname])

    # * To select only a few reqions,
    # * Find all the regions that are not in the list, change the distance value to something absurd.

    if select != 'all':
        grid[np.isin(rlpbc, select, assume_unique=False, invert=True)]=-10
        # faces = faces[np.isin(colours, select, assume_unique=False, invert=False)]
    # * FInd the isosurface
    from skimage.measure import marching_cubes_lewiner as mcl
    verts, faces, normals, values = mcl(grid, value, step_size=1)#, spacing=[0.2,0.2,0.2])

    # * Make a colorscale
    cvals = np.linspace(0,1,32)
    cscale=[[cvals[i],"rgb("+np.str(np.random.randint(0,256))+','+np.str(np.random.randint(0,256))+','+np.str(np.random.randint(0,256))+')'] for i in range(32)]

    #* Find the colour for the vertices
    from scipy.interpolate import RegularGridInterpolator as RGI
    xx = np.linspace(0, rlpbc.shape[0]-1, rlpbc.shape[0])
    yy = np.linspace(0, rlpbc.shape[1]-1, rlpbc.shape[1])
    zz = np.linspace(0, rlpbc.shape[2]-1, rlpbc.shape[2])
    rl = RGI((xx,yy,zz), rlpbc, method='nearest')
    colours = rl(verts)
    import plotly.graph_objects as go
    #* Find the actual coordinates
    hovertext=["Region: "+str(c) for c in colours]
    true_verts = np.dot(A_unit, (verts/rlpbc.shape).T).T
    data_plot = go.Mesh3d(x=true_verts[:,0],y=true_verts[:,1],z=true_verts[:,2],intensity=colours, hoverinfo='all', hovertext=hovertext, flatshading=False, colorscale=cscale , i=faces[:,0],j=faces[:,1],k=faces[:,2], showscale=False, opacity =opacity, name='Pores')
    # data_plot = go.Mesh3d(x=true_verts[:,0],y=true_verts[:,1],z=true_verts[:,2], alphahull=5, intensity=colours.astype(np.int), hoverinfo='all', hovertext=hovertext, flatshading=False, colorscale=cscale, showscale=False, opacity =opacity, name='Pores')
    data_plot.update(               lighting=dict(ambient=0.18,
                                                diffuse=1,
                                                fresnel=0.1,
                                                specular=1,
                                                roughness=0.05,
                                                facenormalsepsilon=1e-15,
                                                vertexnormalsepsilon=1e-15),
                                    lightposition=dict(x=100,
                                                y=200,
                                                z=0
                                                    )
                    )

    return data_plot
def plot_surface_dgrid(dgrid_np, data, value, rlpbc, select='all', opacity=1.0, cscale=None):
    # import h5py
    import numpy as np
    import ase
    from ase.io import read
    import copy 
    dgrid_np_copy = copy.deepcopy(dgrid_np)
    # data = read(path_to_cif)

    A_unit = ase.geometry.complete_cell(data.get_cell())

    # hfile         = h5py.File(path_to_h5)
    # grid      = np.array(hfile[gridname])
    # * To select only a few reqions,
    # * Find all the regions that are not in the list, change the distance value to something absurd.

    
    if select != 'all':
        dgrid_np_copy[np.isin(rlpbc, select, assume_unique=False, invert=True)]=-10

    # * FInd the isosurface
    from skimage.measure import marching_cubes_lewiner as mcl
    verts, faces, normals, values = mcl(dgrid_np_copy, value, step_size=1)#, spacing=[0.2,0.2,0.2])

    if cscale==None:
        # * Make a colorscale if no colorscale is provided
        cvals = np.linspace(0,1, (rlpbc.max()).astype(np.int))
        cscale=[[cvals[i],"rgb("+np.str(np.random.randint(0,256))+','+np.str(np.random.randint(0,256))+','+np.str(np.random.randint(0,256))+')'] for i in range((rlpbc.max()).astype(np.int))]

    #* Find the colour for the vertices
    from scipy.interpolate import RegularGridInterpolator as RGI
    xx = np.linspace(0, rlpbc.shape[0]-1, rlpbc.shape[0])
    yy = np.linspace(0, rlpbc.shape[1]-1, rlpbc.shape[1])
    zz = np.linspace(0, rlpbc.shape[2]-1, rlpbc.shape[2])
    rl = RGI((xx,yy,zz), rlpbc, method='nearest')
    colours = rl(verts)
    import plotly.graph_objects as go
    #* Find the actual coordinates
    hovertext=["Region: "+str(int(c)) for c in colours]
    true_verts = np.dot(A_unit, (verts/rlpbc.shape).T).T
    data_plot = go.Mesh3d(x=true_verts[:,0],y=true_verts[:,1],z=true_verts[:,2],intensity=colours.astype(np.int), hoverinfo='all', hovertext=hovertext, flatshading=False, colorscale=cscale , i=faces[:,0],j=faces[:,1],k=faces[:,2], opacity =opacity, name='Pores')
    data_plot.update(               lighting=dict(ambient=0.18,
                                                diffuse=1,
                                                fresnel=0.1,
                                                specular=1,
                                                roughness=0.05,
                                                facenormalsepsilon=1e-15,
                                                vertexnormalsepsilon=1e-15),
                                    lightposition=dict(x=10,
                                                y=10,
                                                z=30
                                                    ),showscale=False
                    )

    return data_plot
def plot_windows(window_centers, color='blue'):
    import plotly.graph_objects as go
    # import plotly.io as pio
    # pio.renderers.default='jupyterlab'
    # fig=go.Figure()
    data_scatter = go.Scatter3d(x=window_centers[:,0],y=window_centers[:,1],z=window_centers[:,2], mode='markers', marker=dict(color=color, opacity=0.5, size=20), name='Windows')
    # fig.add_trace(data_scatter)
    # fig.update_layout(template='plotly_dark', title='Scatter plot of angles', showlegend=True)
    return data_scatter
def plot_cell(data, color='purple'):
    import plotly.graph_objects as go
    from ase.io import read
    # data = read(t1.file)
    A_unit = data.get_cell().T
    cell_lines = draw_cell(A_unit)
    data_cell = go.Scatter3d(z=cell_lines[:,2],y=cell_lines[:,1],x=cell_lines[:,0], mode='lines', line=dict(color=color, width=8), name='Cell')
    return data_cell
def create_windows_fig_h5(path_to_h5, gridname, t1, rlpbc, wcenters, value):
    # import pyIsoP.grid3D as grid3D
    # path_to_cif='C:\\Users\\arung\\Google Drive\\Current Research\\ZIF-8\\pore_topo\\ZIF-8_2x2x2.cif'
    # t1         = grid3D.grid3D(path_to_cif,spacing=0.2)
    data_surface = plot_surface(path_to_h5, gridname, t1, value, rlpbc)
    data_windows = plot_windows(wcenters)
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(data_surface)
    fig.add_trace(data_windows)
    fig.update_layout(template='plotly_dark', hoverlabel_align ='right', plot_bgcolor='black', showlegend=True, margin=dict(l=10, r=30, t=10, b=10))
    return fig
def create_windows_fig_dgrid(dgrid_np, t1, rlpbc, wcenters, value):
    # import pyIsoP.grid3D as grid3D
    # path_to_cif='C:\\Users\\arung\\Google Drive\\Current Research\\ZIF-8\\pore_topo\\ZIF-8_2x2x2.cif'
    # t1         = grid3D.grid3D(path_to_cif,spacing=0.2)
    data_surface = plot_surface_dgrid(dgrid_np, t1, value, rlpbc)
    data_windows = plot_windows(wcenters)
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_trace(data_surface)
    fig.add_trace(data_windows)
    fig.update_layout(template='plotly_dark', hoverlabel_align ='right', plot_bgcolor='black', showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
    return fig
def draw_cell(A):
    import numpy as np
    avec = A[:,0].T
    bvec = A[:,1].T
    cvec = A[:,2].T
    origin = np.array([0,0,0])
    cell_lines = []
    cell_lines.append(origin)
    cell_lines.append(avec)
    cell_lines.append([None, None, None])
    cell_lines.append(origin)
    cell_lines.append(bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(origin)
    cell_lines.append(cvec)
    cell_lines.append([None, None, None])
    cell_lines.append(avec)
    cell_lines.append(avec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(bvec)
    cell_lines.append(avec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(bvec)
    cell_lines.append(cvec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(cvec)
    cell_lines.append(cvec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(cvec)
    cell_lines.append(cvec+avec)
    cell_lines.append([None, None, None])
    cell_lines.append(avec)
    cell_lines.append(cvec+avec)
    cell_lines.append([None, None, None])
    cell_lines.append(avec+bvec)
    cell_lines.append(cvec+avec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(avec+cvec)
    cell_lines.append(cvec+avec+bvec)
    cell_lines.append([None, None, None])
    cell_lines.append(cvec+bvec)
    cell_lines.append(cvec+avec+bvec)
    cell_lines.append([None, None, None])
    return np.array(cell_lines)
def compute_dgrid(path_to_cif, forcefield, spacing=0.5, epsilon=46, sigma=3.95, compute=True):
    import pyIsoP.grid3D      as grid3D
    import pyIsoP.potentials  as potentials
    import pyIsoP.forcefields as forcefields
    import pyIsoP.writer      as writer

    ####################################################################
    # Set up the grid
    t1         = grid3D.grid3D(path_to_cif,spacing=spacing)          # Intialize grid3D object
    f1         = forcefields.forcefields(t1, forcefield=forcefield, sigma=sigma, epsilon=epsilon)      # Update the force field details to grid obj. t1
    dgrid_dask = grid3D.grid3D.dgrid_calc_dask(t1,f1)

    if compute:
        return [t1, f1, dgrid_dask.compute()]
    else:
        return [t1, f1, dgrid_dask]
def compute_Egrid(path_to_cif, forcefield, spacing=0.5, epsilon=46, sigma=3.95, compute=True):
    import pyIsoP.grid3D      as grid3D
    import pyIsoP.potentials  as potentials
    import pyIsoP.forcefields as forcefields
    import pyIsoP.writer      as writer

    ####################################################################
    # Set up the grid
    t1         = grid3D.grid3D(path_to_cif,spacing=spacing)          # Intialize grid3D object
    f1         = forcefields.forcefields(t1, forcefield=forcefield, sigma=sigma, epsilon=epsilon)      # Update the force field details to grid obj. t1
    grid_dask = grid3D.grid3D.grid_calc_dask(t1,f1)

    if compute:
        return [t1, f1, grid_dask.compute()]
    else:
        return [t1, f1, grid_dask]
def diffusion_paths(local_maxima, connections, field, T = 77.0,  cut_off=1E8):

    import dijkstra3d as d3d
    import numpy as np
    from tqdm import tqdm

    # * Define the scalar field
    # dk_field = egrid
    # dk_field[dk_field>=cut_off]=cut_off
    # dk_field = (dk_field - np.min(egrid))/T +1 # Weight cannot be zero
    # dk_field = dk_field.astype(np.int)
    # field = np.asfortranarray(dk_field) # exp(+\beta*E)
    local_maxima = np.vstack(np.where(local_maxima)).T
    paths = [d3d.dijkstra(field, source=local_maxima[c[0]-1], target=local_maxima[c[1]-1]) for c in tqdm(connections)]
    return paths
def plot_paths(paths, t1, rlpbc, localmaxi):
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats
    from ase.io import read
    data = read(t1.file)
    A_unit = data.get_cell().T
    # fig = go.Figure()

    # * Combine all the paths into one trace to make it easy on plotting.
    master_paths = []
    for p in paths:
        true_points = np.dot(A_unit, (p/rlpbc.shape).T).T
        master_paths.append(true_points)
        master_paths.append(np.array([None, None, None]))
    localmaxi = np.dot(A_unit, (np.vstack(np.where(localmaxi)).T/rlpbc.shape).T).T
    master_paths = np.vstack(master_paths)
    # true_points = np.dot(t1.A, master_paths.T).T
    print(master_paths.shape)
    # print(stats.describe(master_paths!=None))
    data_paths = go.Scatter3d(x=list(master_paths[:,0]), y=list(master_paths[:,1]), z=list(master_paths[:,2]), mode='markers+lines', marker=dict(size=6, color='yellow', opacity=0.8), line=dict(color='red', width=4), name='All Diffusion Paths')
    # fig.add_trace(data_scatter)
    data_maxi = go.Scatter3d(x=localmaxi[:,0], y=localmaxi[:,1], z=localmaxi[:,2], mode='markers', marker=dict(size=12, color='green', opacity=1.0),  name='Local Maxima')
    return data_paths, data_maxi
def dgrid_to_field(dgrid_np):
    import numpy as np
    frame = np.where(dgrid_np<=0)
    dgrid_np = dgrid_np - np.min(dgrid_np)
    dgrid_np = dgrid_np.astype(np.int)
    dgrid_np[frame]=1E-6
    dgrid_np = np.asfortranarray(1/dgrid_np)
    return dgrid_np
def h5_to_field(path_to_h5, gridname):
    import h5py
    import numpy as np

    hfile         = h5py.File(path_to_h5,'r')
    dgrid_np      = np.array(hfile[gridname])
    frame = np.where(dgrid_np<=0)
    dgrid_np = dgrid_np - np.min(dgrid_np)
    dgrid_np = dgrid_np.astype(np.int)
    dgrid_np[frame]=1E-8
    dgrid_np = np.asfortranarray(1/(10*dgrid_np))
    return dgrid_np
def get_default_colors():
    """[summary]
    Uses the ASE's jmol colorscheme as the defual for plotting the atoms.     
    :return: [NOrmalized rgb values of the jmol colorscheme as included within ASE python module. ]
    :rtype: [(N,3) numpy array of floats all within [0,1]]
    """
    from ase.data import colors
    return colors.jmol_colors
def draw_framework(data, colorscheme=get_default_colors(), opacity=[1.0, 1.0]):

    """ Take a grid3D object and plot the atoms and bonds after detecting them using PyMatGen and return
    the Plotly scatter trace to be plotted alongside isosurfaces, windows and diffusion paths.  
    :param t1: [grid3D object that needs to be plotted]
    :type t1: [grid3D object]
    """
    from ase.io import read
    from ase import neighborlist
    # data = read(path_to_cif)
    A_unit = data.get_cell().T
    coord = data.get_positions()
    names = data.get_chemical_symbols()
    def find_covalent_bonds(molecule):
        import itertools
        from pymatgen.core.bonds import CovalentBond
        number_of_sites = len(molecule.sites)
        bonds=[]
        for a,b in itertools.combinations(range(number_of_sites),2):
            if CovalentBond.is_bonded(molecule.sites[a], molecule.sites[b], default_bl=2.0):
                bonds.append([a,b])
        return np.array(bonds)

    import numpy as np
    from pymatgen import Molecule
    print("1. Creating Molecule ...")
    # coord = np.dot(A_unit, (coord+[0.5,0.5,0.5]).T).T
    mol = Molecule(names, coord)
    print("2. Computing bonds ...")
    bonds = find_covalent_bonds(mol)
    # print(bonds)
    # print(np.hstack(bonds))
    edges_master = []
    # bonds_atoms_list = []
    for b in bonds:
        edges_master.append(coord[b[0]])
        edges_master.append(coord[b[1]])
        edges_master.append([None, None, None])

        # * Append the atom id's that are in each bond, None is a spacer.
        # bonds_atoms_list.append(b[0])
        # bonds_atoms_list.append(b[1])
        # bonds_atoms_list.append(None)
        
    edges_master=np.vstack(edges_master)
    # print(bonds_atoms_list)
    print("3. Creating Plotly figure traces")
    # * Figure out how to plot the framework atoms
    import plotly.graph_objects as go
    # atom_colors           = {'C':'teal', 'H':'yellow', 'O':'red', 'N':'blue ', 'Zn':'red', 'Zr': 'pink'} # this line needs to be updated.
    # scatter_colors        = [atom_colors[t] for t in np.array(names)]

    # * Create a color scale using ASE cpk colorlist.
    from ase.data import colors
    from sklearn.preprocessing import minmax_scale
    cols                    = np.round(colorscheme*255, decimals=0).astype(np.int)
    colorlist               = ["rgb("+ ",".join(c.astype(np.str)) +")"  for c in cols]
    # atomic_numbers        = np.vstack(data.get_atomic_numbers())
    # atom_colors           =   minmax_scale(np.linspace(0,103, 104), feature_range=(0,1))
    atom_colors             = np.linspace(0,1,len(cols))
    cscale                  = [[atom_colors[i], colorlist[i]] for i in range(len(colorlist))]

    # * Hovertext for each point
    hovertext               = ['Atom No: '+str(i)+ ', Atom type: '+ names[i] for i in range(len(names))]

    # * Scale the atom sizes beautifully.
    # atom_dia              = f1.sigma_array*2-f1.sigma
    atom_dia                = 2*neighborlist.natural_cutoffs(data)
    print(atom_dia)
    scatter_size             = np.round(minmax_scale(atom_dia, feature_range=(12,30)))

    bond_atoms_diameter     = np.array(atom_dia)[np.hstack(bonds)]
    bond_atoms_Z            = data.get_atomic_numbers()[np.hstack(bonds)] -1 # The indices in the color scale start at 0, not 1.
    bond_scatter_size       = np.round(minmax_scale(bond_atoms_diameter, feature_range=(12,35)))

    data_scatter            = go.Scatter3d(x=coord[:,0] , y=coord[:,1], z=coord[:,2] , mode='markers', hovertext=hovertext, marker=dict(opacity=opacity[0], size=scatter_size, color=atom_colors[data.get_atomic_numbers()-1], colorscale=cscale, sizemode='diameter'),name='framework_atoms')
    data_edges              = go.Scatter3d(x=edges_master[:,0] , y=edges_master[:,1], z=edges_master[:,2] , line=dict(width=15, color='grey'), mode='lines', opacity=opacity[1], name='framework_bonds')
    # data_framework          = go.Scatter3d(x=edges_master[:,0] , y=edges_master[:,1], z=edges_master[:,2] , line=dict(width=15, color='grey'), mode='markers+lines', hovertext=hovertext, marker=dict(opacity=opacity[0], size=bond_scatter_size, color=bond_atoms_Z, colorscale=cscale), opacity=opacity[1], name='Framework')
    return  data_scatter, data_edges
def single_point_energy(point, t1, f1):
    """[summary]
    Compute the distance to the cetner 
    :param point: [Coordinates at which the distance is to be calculated ]
    :type point: [(1x3) array of floats.]
    :param t1: [Grid 3D object containing all the framework information.]
    :type t1: [An instance of the PyIsoP grid3D class]
    :param f1: [Forcefield object containing information about the forcefield.]
    :type f1: [An instance of the PyIsoP forcefield class.]
    :return: [Distance to the center of the nearest framwwork atom to the point.]
    :rtype: [float64]
    """
    import numpy as np
    # Compute the energy at any grid point.
    point_unit = np.dot(t1.A_inv, point.T).T
    dr = point_unit-t1.coord
    # dr = g-t1.coord
    dr = dr-np.round(dr)
    dr = np.dot(t1.A, dr.T).T
    rsq = np.sum(dr**2, axis=1)
    return np.sum((4*f1.eps_array) * ((f1.sigma_array**12/(rsq)**6) - ((f1.sigma_array)**6/(rsq)**3)))
def single_point_distance(point, t1, f1):
    import numpy as np
    # Compute the distance to the nearest framework surface at any grid point.
    point_unit = np.dot(t1.A_inv, point.T).T
    dr = point_unit-t1.coord
    dr = dr-np.round(dr)
    dr = np.dot(t1.A, dr.T).T
    rsq = np.sum(dr**2, axis=1) # * Actual center to center distance squared.
    rsqrt = np.sqrt(rsq) # * The center to center distance
    return np.min((rsqrt-(f1.sigma_array*2-f1.sigma)*0.5)) #* Subtract the diameter of the framework atom.
def compute_dgrid_gpu(path_to_cif, spacing=0.2, chunk_size=5000):


    from ase.io import read
    import numpy as np 
    import dask.array as da 
    import cupy as cp
    import pandas as pd
    from tqdm import tqdm
    import ase

    data    = read(path_to_cif)
    fpoints = cp.asarray(data.get_scaled_positions()-0.5)
    cell = cp.asarray(ase.geometry.complete_cell(data.get_cell()) )
    
    radius_series =  pd.Series({'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96,
            'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66,
            'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
            'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05,
            'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
            'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
            'Mn': 1.50, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24,
            'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20,
            'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
            'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75,
            'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46,
            'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
            'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38,
            'I': 1.39, 'Xe': 1.40, 'Cs': 2.44, 'Ba': 2.15,
            'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01,
            'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
            'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89,
            'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
            'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44,
            'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
            'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40,
            'At': 1.50, 'Rn': 1.50, 'Fr': 2.60, 'Ra': 2.21,
            'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96,
            'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69})
    radii = cp.asarray(radius_series[data.get_chemical_symbols()].values).reshape(-1,1)
    [nx, ny, nz] = (data.get_cell_lengths_and_angles()[0:3]/spacing).astype(np.int)+1
#     print([nx,ny,nz])
    gpoints = da.stack(da.meshgrid(np.linspace(-0.5, 0.5, nx), np.linspace(-0.5, 0.5, ny),np.linspace(-0.5, 0.5, nz), indexing='ij'), -1).reshape(-1, 3).rechunk(chunk_size,3)
    
#     def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
#         points = cp.asarray(points)
#         return cp.min(cp.linalg.norm(cp.dot(cell, (cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0)-cp.around(cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0))).reshape(-1,3).T).T, axis=1).reshape(fpoints.shape[0],-1)-radii, axis=0)
    
    def gpd(points, fpoints=fpoints, cell=cell, radii=radii):
        points = cp.asarray(points)
        
        diff = cp.expand_dims(points, axis=1)-cp.expand_dims(fpoints, axis=0)
        diff = (diff - cp.around(diff)).reshape(-1,3)
        diff = cp.dot(cell, diff.T).T
        diff = cp.linalg.norm(diff, axis=1).reshape(-1,fpoints.shape[0])-radii.T
        return cp.min(diff, axis=1)

    gpoints_da = gpoints.map_blocks(gpd,chunks=(chunk_size,1))
    
    distance_grid = []
    for block in tqdm(gpoints_da.blocks):
        # print("Working on block {} \n".format(i))
        distance_grid.append(block.compute())
        
    #  Free the gpu memeory before returning the output
    del data, cell, fpoints, radius_series, radii
    cp._default_memory_pool.free_all_blocks()
    cp._default_pinned_memory_pool.free_all_blocks()
    return np.hstack(distance_grid).reshape(nx,ny,nz)

    
def read_raspa_pdb(path_to_file):   



    
        """
        created by Arun Gopalan, Snurr Research Group.                   _                                 ___  ___  ___ 
        _ __ ___  __ _  __| |  _ __ __ _ ___ _ __   __ _    / _ \/   \/ __\
        | '__/ _ \/ _` |/ _` | | '__/ _` / __| '_ \ / _` |  / /_)/ /\ /__\//
        | | |  __/ (_| | (_| | | | | (_| \__ \ |_) | (_| | / ___/ /_// \/  \
        |_|  \___|\__,_|\__,_| |_|  \__,_|___/ .__/ \__,_| \/  /___,'\_____/
                                             |_|                            
        Reads the output PDB movie file from RASPA into separate configurations, including the chemical symbols and
        cell dimensions. 
        :type path_to_file: str
        :param path_to_file: path to the RASPA PDB file
    
        :raises:
        Not sure.
    
        :rtype: a python dictionary with 'cells', 'symbols' and 'coord' which stand for cell dimensions, printed symbols and coordinates
        """
    
        import numpy as np

        f = open(path_to_file).readlines()
        start = np.where(["MODEL" in line for line in f ])[0] + 2  # * Start of config 
        ends = np.where(["ENDMDL" in line for line in f ])[0]   # End for config
        cryst = np.where(["CRYST" in line for line in f ])[0]   # box shape for the config

        data = [f[start[i]:ends[i]] for i in range(len(start))]

        coord = np.array([[np.array(line.split()[4:7]).astype(np.float) for line in d ] for d in data])
        cell_dims = np.array([np.array(line.split())[1:].astype(np.float)  for line in  f if "CRYST" in line])
        symbols = np.array([[np.array(line.split()[2]) for line in d ] for d in data])

        output = {}
        output['cells']=cell_dims
        output['coord']=coord
        output['symbols']=symbols

        return output

def plot_united_alkane(coord, nbeads, bead_color, bond_color, legend_str):

    # make the necessary imports 
    import numpy as np
    import plotly.graph_objects as go

    coord_adj = np.insert(coord, obj=range(nbeads, len(coord), nbeads), values= [None, None, None], axis=0)
    trace_alkanes = go.Scatter3D(x=coord_adj[:,0],y=coord_adj[:,1],z=coord_adj[:,2],mode='markers+lines', marker=dict(size=12, color=bead_color,  opacity=0.8), line=dict(color=bond_color, width=6), name=legend_str)   
    
    return trace_alkanes





