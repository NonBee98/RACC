import json
import os

import exiftool
import numpy as np

def read_exif_from_dng(raw_file):
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(raw_file)[0]
    cfa_pattern = list(map(int, metadata['EXIF:CFAPattern2'].split()))
    wp = np.array(list(map(float, metadata['EXIF:AsShotNeutral'].split())))
    orientation = metadata['EXIF:Orientation']
    cam_calibration = list(
        map(float, metadata['EXIF:CameraCalibration1'].split()))
    cm_65 = list(map(float, metadata['EXIF:ColorMatrix1'].split()))
    cm_65 = np.array(cm_65).reshape((3, 3))
    cm_28 = list(map(float, metadata['EXIF:ColorMatrix2'].split()))
    cm_28 = np.array(cm_28).reshape((3, 3))

    fm_65 = list(map(float, metadata['EXIF:ForwardMatrix1'].split()))
    fm_65 = np.array(fm_65).reshape((3, 3))
    fm_28 = list(map(float, metadata['EXIF:ForwardMatrix1'].split()))
    fm_28 = np.array(fm_28).reshape((3, 3))

    ret = {
        'cfa_pattern': cfa_pattern,
        'orientation': orientation,
        'cam_calibration': cam_calibration,
        'wp': wp,
        'cm_65': cm_65,
        'cm_28': cm_28,
        'fm_65': fm_65,
        'fm_28': fm_28
    }
    return ret


def extract_exif(file: str) -> dict:
    path = os.path.abspath(file)
    cmd = "exiftool -a -u -g1 -j {}".format(path)
    result = os.popen(cmd)
    context = result.read()
    exif_tool_ret = json.loads(context)[0]
    result.close()
    ColorMatrix1 = np.array(
        list(map(float,
                 exif_tool_ret['IFD0']['ColorMatrix1'].split()))).reshape(
                     (3, 3))
    ColorMatrix2 = np.array(
        list(map(float,
                 exif_tool_ret['IFD0']['ColorMatrix2'].split()))).reshape(
                     (3, 3))
    ForwardMatrix1 = np.array(
        list(map(float,
                 exif_tool_ret['IFD0']['ForwardMatrix1'].split()))).reshape(
                     (3, 3))
    ForwardMatrix2 = np.array(
        list(map(float,
                 exif_tool_ret['IFD0']['ForwardMatrix2'].split()))).reshape(
                     (3, 3))
    try:
        CameraCalibration1 = np.array(
            list(
                map(float, exif_tool_ret['IFD0']
                    ['CameraCalibration1'].split()))).reshape((3, 3))
    except:
        CameraCalibration1 = np.eye(3)
    try:
        CameraCalibration2 = np.array(
            list(
                map(float, exif_tool_ret['IFD0']
                    ['CameraCalibration2'].split()))).reshape((3, 3))
    except:
        CameraCalibration2 = np.eye(3)
    AsShotNeutral = np.array(
        list(map(float, exif_tool_ret['IFD0']['AsShotNeutral'].split())))
    CalibrationIlluminant1 = exif_tool_ret['IFD0']['CalibrationIlluminant1']
    CalibrationIlluminant2 = exif_tool_ret['IFD0']['CalibrationIlluminant2']
    Orientation = exif_tool_ret['IFD0']['Orientation']
    BlackLevel = np.array(
        list(map(float, exif_tool_ret['SubIFD']['BlackLevel'].split())))
    WhiteLevel = int(exif_tool_ret['SubIFD']['WhiteLevel'])
    CFAPattern2 = list(map(int,
                           exif_tool_ret['SubIFD']['CFAPattern2'].split()))

    ImageWidth = int(exif_tool_ret['SubIFD']['ImageWidth'])
    ImageHeight = int(exif_tool_ret['SubIFD']['ImageHeight'])
    try:
        DefaultCropOrigin = np.array(
            list(map(int,
                     exif_tool_ret['SubIFD']['DefaultCropOrigin'].split())))
    except:
        DefaultCropOrigin = np.zeros(2)
    try:
        DefaultCropSize = np.array(
            list(map(int, exif_tool_ret['SubIFD']['DefaultCropSize'].split())))
    except:
        DefaultCropSize = np.array([ImageWidth, ImageHeight])
    exif = {
        "ColorMatrix1": ColorMatrix1,
        "ColorMatrix2": ColorMatrix2,
        "ForwardMatrix1": ForwardMatrix1,
        "ForwardMatrix2": ForwardMatrix2,
        "CameraCalibration1": CameraCalibration1,
        "CameraCalibration2": CameraCalibration2,
        "AsShotNeutral": AsShotNeutral,
        "Orientation": Orientation,
        "CalibrationIlluminant1": CalibrationIlluminant1,
        "CalibrationIlluminant2": CalibrationIlluminant2,
        "BlackLevel": BlackLevel,
        "WhiteLevel": WhiteLevel,
        "CFAPattern2": CFAPattern2,
        "ImageWidth": ImageWidth,
        "ImageHeight": ImageHeight,
        "DefaultCropOrigin": DefaultCropOrigin,
        "DefaultCropSize": DefaultCropSize,
    }
    return exif
