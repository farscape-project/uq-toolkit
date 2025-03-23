"""
This module provides functions to help automate the adjustment of the mesh of a
HIVE induction heating coil and heating target, including adjustments to their
relative position and meshing of the void region around them.
"""
import sys
import subprocess
import hjson
from spython.main import Client # singularity for running vacuum mesher


def coil_target_distance(coil_bottom_surface_id, target_top_surface_id):
    """
    Find minimum distance between the coil and the target.
    Returns a tuple containing the minimum distance, and the offset vector
    between the two nearest points.
    """

    # Get distance and offset between surfaces at closest point
    dist_info = cubit.measure_between_entities(
        "surface", coil_bottom_surface_id, "surface", target_top_surface_id)
    dist = dist_info[0]
    coil_point_closest_to_target = (dist_info[1], dist_info[2], dist_info[3])
    target_point_closest_to_coil = (dist_info[4], dist_info[5], dist_info[6])
    # Calculate vector from coil to target at closest points.
    offset_vector = tuple(
        target_point_closest_to_coil[i] - coil_point_closest_to_target[i]
        for i in range(3))
    return dist, offset_vector


def align_coil_to_target(
        coil_volume_id, coil_bottom_surface_id, target_top_surface_id,
        coil_target_offset_vector):
    """
    Align coil volume to target, given a defined offset vector between the
    closest points on the coil and target.

    Args:
    coil_volume_id (int): Volume ID of the coil to move.
    coil_bottom_surface_id (int): Surface ID of the (bottom) surface of the
                                  coil.
    target_top_surface_id (int): Surface ID of the (top) surface of the target.
    coil_target_offset_vector (tup): Offset vector as a (dx, dy, dz) tuple of
                                     x, y, z offsets between the closest
                                     points on the coil and target.
    """

    # Calculate the initial offset vector between the two nearest points on
    # the coil and target.
    _, offset_vector = coil_target_distance(
        coil_bottom_surface_id, target_top_surface_id)
    # Calculate the translation vector to move our coil to the desired new
    # position, from the old one
    translation_vector = tuple(
        offset_vector[i] + coil_target_offset_vector[i] for i in range(3))
    # Set up args required for the cubit move command
    coil_move_args = {"coil_vid": coil_volume_id,
                      "x_shift": translation_vector[0],
                      "y_shift": translation_vector[1],
                      "z_shift": translation_vector[2]}
    # Move coil
    cubit.cmd(
        'move Volume %(coil_vid)d x %(x_shift)f y %(y_shift)f z %(z_shift)f'
        'include_merged ' %
        coil_move_args)


def automesh_solids(solid_volume_ids, size_auto_factor):
    """
    Meshes set of solid volumes to an automatically determined mesh size.

    solid_volume_ids: Iterable of ints representing IDs
                      of solid volumes to mesh.
    size_auto_factor: Size factor (int) for Cubit to determine mesh size.
                      Smaller values result in finer meshes.
    """

    # Iterate over all solid volumes in solid_volume_ids
    for vid in solid_volume_ids:
        mesh_args = {"vid": vid, "size_auto_factor": size_auto_factor}
        # # Delete existing mesh
        # cubit.cmd(
        #     'delete mesh volume %(vid)d' % mesh_args)
        # Set desired mesh size of volume
        cubit.cmd(
            f'volume {vid} size auto factor {size_auto_factor}')
        # Mesh volume
        cubit.cmd(f'mesh volume {vid}')


def export_solids_to_exodus(output_solidmesh_path, output_cub5_path):
    """
    Export solid mesh to exodus and save the current .cub5 file.
    """

    cubit.cmd('set exodus netcdf4 off')
    cubit.cmd('set large exodus file on')
    # Export solid mesh to Exodus
    cubit.cmd('export mesh "%(exodus_solid_mesh)s"  overwrite ' %
              {"exodus_solid_mesh": output_solidmesh_path})
    # Save updated .cub5 file
    cubit.cmd('save cub5 "%(solid_cub5)s" overwrite journal' %
              {"solid_cub5": output_cub5_path})


def export_target_mesh_to_exodus(output_targetmesh_path, target_block):
    """
    Export solid mesh to exodus and save the current .cub5 file.
    """
    cubit.cmd('Sideset 4 ADD Surface 17')
    cubit.cmd('Sideset 4 Name "inner_pipe"')
    cubit.cmd('set exodus netcdf4 off')
    cubit.cmd('set large exodus file on')
    # Export solid mesh of target to Exodus
    cubit.cmd(
        'export mesh "%(exodus_solid_mesh)s" '
        'block %(target_block)d overwrite ' % {
            "exodus_solid_mesh": output_targetmesh_path,
            "target_block": target_block})


def mesh_void(vacuummesher_img_path,
              vacuummesher_exe_path,
              input_solid_mesh_path,
              output_vacuum_mesh_path,
              bound_len,
              sideset_one_id,
              sideset_two_id,
              max_tri,
              max_tet):
    """
    Add a void mesh to an input solid mesh using Bill Ellis's
    VacuumMesher tool.

    Intended for use with open coils, which have coil terminals that intersect
    the boundary of the computational domain. Uses fTetWild to generate
    tetrahedra in the void mesh.

    Args:
    vacuummesher_img_path (str): path to VacuumMesher singularity image (SIF file).
    vacuummesher_exe_path (str): path to VacuumMesher's coilVacuum executable.
    input_solid_mesh_path (str): path to Exodus file representing solid mesh
                                 void mesh is to be added to.
    output_vacuum_mesh_path (str): path to file to be created to store the new
                                   Exodus mesh containing void + solid meshed
                                   volume.
    bound_len (double) : side length (in m) of cubic volume to generate void
                         around solids.
    sideset_one_id (int): sideset of coil input terminal.
                          Needed to identify terminal plane.
    sideset_two_id (int): sideset of coil output terminal.
                          Needed to identify terminal plane.
    max_tri (double): maximum size of triangles generated in void.
    max_tet (double): maximum size of tetrahedra generated in void.
    """

    # Call the external VacuumMesher tool coilVacuum to generate the void mesh.
    Client.execute(
        vacuummesher_img_path,
        [vacuummesher_exe_path,
         "-i", input_solid_mesh_path,
         "-o", output_vacuum_mesh_path,
         "--bound_len", str(bound_len),
         "--sideset_one_id", str(sideset_one_id),
         "--sideset_two_id", str(sideset_two_id),
         "--max_tri", str(max_tri),
         "--max_tet", str(max_tet),
         "-v", "-v", "-v"]
    )


if __name__ == "__main__":

    with open("mesh_config.jsonc", "r") as f:
        config = hjson.load(f)

    sys.path.append(config["cubit-bin-path"])
    # from container, so we can hard-code the path to the executable
    VACUUMMESHER_PATH = '/opt/VacuumMesher/build/bin/coilVacuum' 

    INPUT_CUB5 = '"./meshed_oval_coil_and_solid_target_in.cub5"'
    OUTPUT_CUB5 = r'./meshed_oval_coil_and_stc_out.cub5'

    OUTPUT_SOLIDMESH = r'./solid_meshed_oval_coil_and_stc.e'
    OUTPUT_TARGETMESH = r'./solid_meshed_stc.e'
    OUTPUT_VACMESH = r'./vac_meshed_oval_coil_and_stc.e'

    COIL_VOLUME_ID = config["coil-volume-id"]
    TARGET_VOLUME_ID = config["target-volume-id"]
    COIL_BOTTOM_SURFACE_ID = config["coil-bottom-surface-id"]
    TARGET_TOP_SURFACE_ID = config["target-top-surface-id"]
    COIL_TARGET_OFFSET_VECTOR = (
        config["coil_target_offset_x"],
        config["coil_target_offset_y"],
        config["coil_target_offset_z"]
    )

    import cubit

    cubit.cmd('open ' + INPUT_CUB5)

    align_coil_to_target(COIL_VOLUME_ID, COIL_BOTTOM_SURFACE_ID,
                         TARGET_TOP_SURFACE_ID, COIL_TARGET_OFFSET_VECTOR)
    print("Aligned coil to target.")

    if config["remesh"]:
        MESH_SIZE_AUTO_FACTOR = 2
        automesh_solids([COIL_VOLUME_ID, TARGET_VOLUME_ID], MESH_SIZE_AUTO_FACTOR)
        print("Solid bodies meshed.")

    export_solids_to_exodus(OUTPUT_SOLIDMESH, OUTPUT_CUB5)
    export_target_mesh_to_exodus(OUTPUT_TARGETMESH, 2)
    print("Exported solid body meshes to Exodus.")

    mesh_void(config["VacuumMesher-image"], VACUUMMESHER_PATH, OUTPUT_SOLIDMESH,
              OUTPUT_VACMESH, 0.3, 1, 2, 3e-04, 1.0)
    print("Meshed vacuum region around solid mesh.")
