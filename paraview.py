# Publication-Quality ParaView Visualization
# Volume with Isosurface blend mode + Slices

import os
import math
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

CONFIG = {
    # Data file - UPDATE THIS
    'data_file': 'C:\\Users\\thyss\\Documents\\University\\Semester 4\\HYCO-PhiFlow\\results\\paraview_spacetime\\train\\real_spacetime.vti',
    
    # Output
    'view_size': [3000, 2000],
    'output_filename': 'paraview_publication.png',
    
    # Rendering
    'samples_per_pixel': 128,
    
    # Camera
    'camera_azimuth': -135,
    'camera_elevation': 20,
    'camera_distance': 500,
    'data_center': [63.5, 63.5, 49.5],
    
    # Slices
    'slice_front_z': 0.0,
    'slice_back_z': 99.0,
    
    # Background - white
    'background': [1.0, 1.0, 1.0],
    
    # Data range
    'data_range': [-3.0, 3.0],
    
    # ISOSURFACE VALUES for volume BlendMode='Isosurface'
    'isosurface_values': [-3.0, 3.0, -2.0, 2.0, -1.0, 1.0],
    
    # Bounding box
    'show_outline': True,
    'outline_color': [0.6, 0.6, 0.6],
    'outline_width': 0.5,
    
    # Show/hide elements
    'show_front_slice': True,
    'show_back_slice': True,
    'show_volume': True,
}

# ----------------------------------------------------------------
# Compute camera position
# ----------------------------------------------------------------

def compute_camera(center, azimuth_deg, elevation_deg, distance):
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    x = center[0] + distance * math.cos(el) * math.sin(az)
    y = center[1] + distance * math.sin(el)
    z = center[2] + distance * math.cos(el) * math.cos(az)
    
    view_up = [
        -math.sin(el) * math.sin(az),
        math.cos(el),
        -math.sin(el) * math.cos(az),
    ]
    
    return [x, y, z], view_up

camera_pos, camera_up = compute_camera(
    CONFIG['data_center'],
    CONFIG['camera_azimuth'],
    CONFIG['camera_elevation'],
    CONFIG['camera_distance']
)

# ----------------------------------------------------------------
# Setup render view
# ----------------------------------------------------------------

materialLibrary1 = GetMaterialLibrary()

renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=CONFIG['view_size'],
    AxesGrid='Grid Axes 3D Actor',
    
    # Camera
    CenterOfRotation=CONFIG['data_center'],
    CameraPosition=camera_pos,
    CameraFocalPoint=CONFIG['data_center'],
    CameraViewUp=camera_up,
    CameraFocalDisk=1.0,
    CameraParallelScale=84.75,
    CameraParallelProjection=1,
    
    # Ray tracing
    EnableRayTracing=1,
    BackEnd='OSPRay pathtracer',
    SamplesPerPixel=CONFIG['samples_per_pixel'],
    LightScale=1.6,
    
    # White background
    Backgroundmode='Backplate',
    EnvironmentalBG=CONFIG['background'],
    
    OSPRayMaterialLibrary=materialLibrary1,
    OrientationAxesVisibility=0,
)

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------

real_spacetime = XMLImageDataReader(
    registrationName='real_spacetime.vti',
    FileName=[CONFIG['data_file']]
)
real_spacetime.Set(
    PointArrayStatus=['smoke'],
    TimeArray='None',
)

data_min, data_max = CONFIG['data_range']

# ----------------------------------------------------------------
# Create slices
# ----------------------------------------------------------------

# Front slice (t=0)
if CONFIG['show_front_slice']:
    slice_front = Slice(registrationName='SliceFront', Input=real_spacetime)
    slice_front.Set(
        SliceType='Plane',
        SliceOffsetValues=[0.0],
        PointMergeMethod='Uniform Binning',
    )
    slice_front.SliceType.Set(
        Origin=[63.5, 63.5, CONFIG['slice_front_z']],
        Normal=[0.0, 0.0, 1.0],
    )

# Back slice (t=99)
if CONFIG['show_back_slice']:
    slice_back = Slice(registrationName='SliceBack', Input=real_spacetime)
    slice_back.Set(
        SliceType='Plane',
        SliceOffsetValues=[0.0],
        PointMergeMethod='Uniform Binning',
    )
    slice_back.SliceType.Set(
        Origin=[63.5, 63.5, CONFIG['slice_back_z']],
        Normal=[0.0, 0.0, 1.0],
    )

# Bounding box
if CONFIG['show_outline']:
    outline = Outline(registrationName='Outline', Input=real_spacetime)

# ----------------------------------------------------------------
# Shared color/opacity for front slice (with opacity mapping)
# ----------------------------------------------------------------

smokeTF2D = GetTransferFunction2D('smoke')
smokeTF2D.Set(
    ScalarRangeInitialized=1,
    Range=[data_min, data_max, 0.0, 1.0],
)

smokeLUT = GetColorTransferFunction('smoke')
smokeLUT.Set(
    TransferFunction2D=smokeTF2D,
    RGBPoints=GenerateRGBPoints(
        preset_name='Cool to Warm (Extended)',
        range_min=data_min,
        range_max=data_max,
    ),
    ColorSpace='Lab',
    NanColor=[0.25, 0.0, 0.0],
    ScalarRangeInitialized=1.0,
    EnableOpacityMapping=1,  # Enables opacity mapping for surfaces
)

smokePWF = GetOpacityTransferFunction('smoke')
smokePWF.Set(
    Points=[
        data_min,  1.0, 0.5, 0.0,
        -0.72,     0.6, 0.5, 0.0,
        -0.16,     0.0, 0.5, 0.0,
        0.15,      0.6, 0.5, 0.0,
        data_max,  1.0, 0.5, 0.0,
    ],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# Display: Volume with Isosurface blend mode
# ----------------------------------------------------------------

if CONFIG['show_volume']:
    volumeDisplay = Show(real_spacetime, renderView1, 'UniformGridRepresentation')
    
    # Separate 2D transfer function for volume
    separate_volumeTF2D = GetTransferFunction2D('smoke', volumeDisplay, separate=True)
    separate_volumeTF2D.Set(
        ScalarRangeInitialized=1,
        Range=[data_min, data_max, 0.0, 1.0],
    )
    
    # Separate color map for volume
    separate_volumeLUT = GetColorTransferFunction('smoke', volumeDisplay, separate=True)
    separate_volumeLUT.Set(
        TransferFunction2D=separate_volumeTF2D,
        RGBPoints=GenerateRGBPoints(
            preset_name='Cool to Warm (Extended)',
            range_min=data_min,
            range_max=data_max,
        ),
        ColorSpace='Lab',
        NanColor=[0.25, 0.0, 0.0],
        ScalarRangeInitialized=1.0,
    )
    
    # Separate opacity for volume
    separate_volumePWF = GetOpacityTransferFunction('smoke', volumeDisplay, separate=True)
    separate_volumePWF.Set(
        Points=[
            data_min,  0.8, 0.5, 0.0,
            -0.4,      0.0,  0.5, 0.0,
            0.4,       0.0,  0.5, 0.0,
            data_max,  0.8, 0.5, 0.0,
        ],
        ScalarRangeInitialized=1,
    )

    
    
    volumeDisplay.Set(
        Representation='Volume',
        ColorArrayName=['POINTS', 'smoke'],
        LookupTable=separate_volumeLUT,
        ScalarOpacityUnitDistance=3.0,
        ScalarOpacityFunction=separate_volumePWF,
        TransferFunction2D=separate_volumeTF2D,
        VolumeRenderingMode='OSPRay Based',
        Shade=0,     # Controls how it reacts to direct light
        UseSeparateColorMap=True,
        
        # KEY: Isosurface blend mode
        BlendMode='Composite',
        # IsosurfaceValues=CONFIG['isosurface_values'],
    )
    
    volumeDisplay.ScaleTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]
    volumeDisplay.OpacityTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]
    volumeDisplay.SliceFunction.Origin = CONFIG['data_center']

# ----------------------------------------------------------------
# Display: Front slice (with opacity mapping)
# ----------------------------------------------------------------

if CONFIG['show_front_slice']:
    slice_frontDisplay = Show(slice_front, renderView1, 'GeometryRepresentation')
    slice_frontDisplay.Set(
        Representation='Surface',
        ColorArrayName=['POINTS', 'smoke'],
        LookupTable=smokeLUT,  # Uses shared LUT with EnableOpacityMapping=1
    )
    slice_frontDisplay.ScaleTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]
    slice_frontDisplay.OpacityTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# Display: Back slice (solid, no opacity mapping)
# ----------------------------------------------------------------

if CONFIG['show_back_slice']:
    slice_backDisplay = Show(slice_back, renderView1, 'GeometryRepresentation')
    
    # Separate 2D TF for back slice
    separate_backTF2D = GetTransferFunction2D('smoke', slice_backDisplay, separate=True)
    separate_backTF2D.Set(
        ScalarRangeInitialized=1,
        Range=[data_min, data_max, 0.0, 1.0],
    )
    
    # Separate color map (no opacity mapping)
    separate_backLUT = GetColorTransferFunction('smoke', slice_backDisplay, separate=True)
    separate_backLUT.Set(
        TransferFunction2D=separate_backTF2D,
        RGBPoints=GenerateRGBPoints(
            preset_name='Cool to Warm (Extended)',
            range_min=data_min,
            range_max=data_max,
        ),
        ColorSpace='Lab',
        NanColor=[0.25, 0.0, 0.0],
        ScalarRangeInitialized=1.0,
    )
    
    # Solid opacity for back slice
    separate_backPWF = GetOpacityTransferFunction('smoke', slice_backDisplay, separate=True)
    separate_backPWF.Set(
        Points=[data_min, 1.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0],
        ScalarRangeInitialized=1,
    )
    
    slice_backDisplay.Set(
        Representation='Surface',
        ColorArrayName=['POINTS', 'smoke'],
        LookupTable=separate_backLUT,
        UseSeparateColorMap=True,
    )
    slice_backDisplay.ScaleTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]
    slice_backDisplay.OpacityTransferFunction.Points = [data_min, 0.0, 0.5, 0.0, data_max, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# Display: Bounding box
# ----------------------------------------------------------------

if CONFIG['show_outline']:
    outlineDisplay = Show(outline, renderView1, 'GeometryRepresentation')
    outlineDisplay.Set(
        Representation='Wireframe',
        ColorArrayName=[None, ''],
        AmbientColor=CONFIG['outline_color'],
        DiffuseColor=CONFIG['outline_color'],
        LineWidth=CONFIG['outline_width'],
        Ambient=1.0,
        Diffuse=0.0,
    )

# ----------------------------------------------------------------
# Color bar
# ----------------------------------------------------------------

smokeLUTColorBar = GetScalarBar(smokeLUT, renderView1)
smokeLUTColorBar.Set(
    WindowLocation='Any Location',
    Position=[0.88, 0.25],
    Title='Density',
    ComponentTitle='',
    TitleFontSize=18,
    TitleBold=1,
    TitleColor=[0.0, 0.0, 0.0],
    LabelFontSize=14,
    LabelColor=[0.0, 0.0, 0.0],
    LabelFormat='%.1f',
    RangeLabelFormat='%.1f',
    AutomaticLabelFormat=0,
    AddRangeLabels=1,
    ScalarBarThickness=20,
    ScalarBarLength=0.45,
    DrawBackground=0,
    DrawScalarBarOutline=1,
    ScalarBarOutlineColor=[0.5, 0.5, 0.5],
)
smokeLUTColorBar.Visibility = 1

if CONFIG['show_front_slice']:
    slice_frontDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# Animation
# ----------------------------------------------------------------

timeKeeper1 = GetTimeKeeper()
timeAnimationCue1 = GetTimeTrack()
animationScene1 = GetAnimationScene()
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)



# ----------------------------------------------------------------
# Catalyst
# ----------------------------------------------------------------

from paraview import catalyst
options = catalyst.Options()

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

if __name__ == '__main__':
    SaveExtractsUsingCatalystOptions(options)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, CONFIG['output_filename'])
    
    Render(renderView1)
    SaveScreenshot(output_path, renderView1, ImageResolution=CONFIG['view_size'])
    
    print(f"Saved: {output_path}")
    print(f"Isosurface values: {CONFIG['isosurface_values']}")