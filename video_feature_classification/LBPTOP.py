import numpy as np

def calc_LBPTOP(VolData, FxRadius, FyRadius, TInterval, NeighborPoints, TimeLength, BorderLength, bBilinearInterpolation, Bincount, Code, landmarks, border):
    #  This function is to compute the LBP-TOP features for a video sequence
    #  Reference:
    #  Guoying Zhao, Matti Pietikainen, "Dynamic texture recognition using local binary patterns
    #  with an application to facial expressions," IEEE Transactions on Pattern Analysis and Machine
    #  Intelligence, 2007, 29(6):915-928.
    #
    #   Copyright 2009 by Guoying Zhao & Matti Pietikainen
    #   Matlab version was Created by Xiaohua Huang
    #  If you have any problem, please feel free to contact guoying zhao or Xiaohua Huang.
    # huang.xiaohua@ee.oulu.fi
    ## ########################################################################
    #  Function: Running this funciton each time to compute the LBP-TOP distribution of one video sequence.
    #
    #  Inputs:
    #
    #  "VolData" keeps the grey level of all the pixels in sequences with [height][width][Length];
    #	   please note, all the images in one sequnces should have same size (height and weight).
    #	   But they don't have to be same for different sequences.
    #
    #  "FxRadius", "FyRadius" and "TInterval" are the radii parameter along X, Y and T axis; They can be 1, 2, 3 and 4. "1" and "3" are recommended.
    #  Pay attention to "TInterval". "TInterval * 2 + 1" should be smaller than the length of the input sequence "Length". For example, if one sequence includes seven frames, and you set TInterval to three, only the pixels in the frame 4 would be considered as central pixel and computed to get the LBP-TOP feature.
    #
    #
    #  "NeighborPoints" is the number of the neighboring points
    #	  in XY plane, XT plane and YT plane; They can be 4, 8, 16 and 24. "8"
    #	  is a good option. For example, NeighborPoints = [8 8 8];
    #
    #  "TimeLength" and "BoderLength" are the parameters for bodering parts in time and space which would not
    #	  be computed for features. Usually they are same to TInterval and the bigger one of "FxRadius" and "FyRadius";
    #
    #  "bBilinearInterpolation": if use bilinear interpolation for computing a neighboring point in a circle: 1 (yes), 0 (no).
    #
    #  "Bincount": For example, if XYNeighborPoints = XTNeighborPoints = YTNeighborPoints = 8, you can set "Bincount" as "0" if you want to use basic LBP, or set "Bincount" as 59 if using uniform pattern of LBP,
    #			  If the number of Neighboring points is different than 8, you need to change it accordingly as well as change the above "Code".
    #  "Code": only when Bincount is 59, uniform code is used.
    #  Output:
    #
    #  "Histogram": keeps LBP-TOP distribution of all the pixels in the current frame with [3][dim];
    #	  here, "3" deote the three planes of LBP-TOP, i.e., XY, XZ and YZ planes.
    #	  Each value of Histogram[i][j] is between [0,1]

    ##
    height, width, Length = VolData.shape

    XYNeighborPoints = NeighborPoints[0];
    XTNeighborPoints = NeighborPoints[1];
    YTNeighborPoints = NeighborPoints[2];

    if (Bincount == 0):
        # normal code
        nDim = 2**(YTNeighborPoints);
        Histogram = np.zeros((3, nDim));
    else:
        # uniform code
        Histogram = np.zeros((3, Bincount)); # Bincount = 59;

    if (bBilinearInterpolation == 0):

        for i in range(TimeLength, Length - TimeLength):

            for yc in range(BorderLength, height - BorderLength):

                for xc in range(BorderLength, width - BorderLength):

                    CenterVal = VolData(yc, xc, i);
                    ## In XY plane
                    BasicLBP = 0;
                    FeaBin = 0;

                    for p in range(0, XYNeighborPoints - 1):
                        X = np.floor(xc + FxRadius * np.cos((2 * np.pi * p) / XYNeighborPoints) + 0.5);
                        Y = np.floor(yc - FyRadius * np.sin((2 * np.pi * p) / XYNeighborPoints) + 0.5);

                        CurrentVal = VolData(Y, X, i);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ^ FeaBin;

                        FeaBin = FeaBin + 1;

                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[1, BasicLBP + 1] = Histogram(1, BasicLBP + 1) + 1;
                    else:
                        Histogram[1, Code(BasicLBP + 1, 2) + 1] = Histogram(1, Code(BasicLBP + 1, 2) + 1) + 1;


                    ## In XT plane
                    BasicLBP = 0;
                    FeaBin = 0;
                    for p in range(0, XTNeighborPoints - 1):
                        X = np.floor(xc + FxRadius * np.cos((2 * np.pi * p) / XTNeighborPoints) + 0.5);
                        Z = np.floor(i + TInterval * np.sin((2 * np.pi * p) / XTNeighborPoints) + 0.5);

                        CurrentVal = VolData(yc, X, Z);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ^ FeaBin;

                        FeaBin = FeaBin + 1;


                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[2, BasicLBP + 1] = Histogram(2, BasicLBP + 1) + 1;
                    else: # uniform patterns
                        Histogram[2, Code(BasicLBP + 1, 2) + 1] = Histogram(2, Code(BasicLBP + 1, 2) + 1) + 1;


                    ## In YT plane
                    BasicLBP = 0;
                    FeaBin = 0;
                    for p in range(0, YTNeighborPoints - 1):
                        Y = np.floor(yc - FyRadius * np.sin((2 * np.pi * p) / YTNeighborPoints) + 0.5);
                        Z = np.floor(i + TInterval * np.cos((2 * np.pi * p) / YTNeighborPoints) + 0.5);

                        CurrentVal = VolData(Y, xc, Z);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ^ FeaBin;

                        FeaBin = FeaBin + 1;

                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[3, BasicLBP + 1] = Histogram(3, BasicLBP + 1) + 1;
                    else:
                        Histogram[3, Code(BasicLBP + 1, 2) + 1] = Histogram(3, Code(BasicLBP + 1, 2) + 1) + 1;




    else: # bilinear interpolation
        for i in range(TimeLength, Length - TimeLength):

            #for yc in range(BorderLength, height - BorderLength - 1, 10):

                #for xc in range(BorderLength, width - BorderLength - 1, 10):
            for lm in landmarks:
                xc, yc = int(lm[0]+border), int(lm[1]+border)
                if 1:
                    CenterVal = VolData[yc, xc, i];
                    ## In XY plane
                    BasicLBP = 0;
                    FeaBin = 0;
                    for p in range(0, XYNeighborPoints - 1):

                        # bilinear interpolation
                        x1 = np.single(xc + FxRadius * np.cos((2 * np.pi * p) / XYNeighborPoints));##"float" are called "single" in Matlab
                        y1 = np.single(yc - FyRadius * np.sin((2 * np.pi * p) / XYNeighborPoints));

                        u = x1 - np.floor(x1-1);
                        v = y1 - np.floor(y1-1);
                        ltx = np.floor(x1-1);
                        lty = np.floor(y1-1);
                        lbx = np.floor(x1-1);
                        lby = np.ceil(y1-1);
                        rtx = np.ceil(x1-1);
                        rty = np.floor(y1-1);
                        rbx = np.ceil(x1-1);
                        rby = np.ceil(y1-1);
                        # the values of neighbors that do not fall exactly on
                        # pixels are estimated by bilinear interpolation of
                        # four corner points near to it.
                        CurrentVal = np.floor(VolData[int(lty), int(ltx), i] * (1 - u) * (1 - v) + VolData[int(lby), int(lbx), i] * (1 - u) * v + VolData[int(rty), int(rtx), i] * u * (1 - v) + VolData[int(rby), int(rbx), i] * u * v);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ** FeaBin;

                        FeaBin = FeaBin + 1;

                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[0, BasicLBP + 1] = Histogram[0, BasicLBP + 1] + 1;
                    else:
                        Histogram[0, Code[BasicLBP + 1, 2] + 1] = Histogram[0, Code[BasicLBP + 1, 2] + 1] + 1;


                    ## In XT plane
                    BasicLBP = 0;
                    FeaBin = 0;
                    for p in range(0, XTNeighborPoints - 1):
                        # bilinear interpolation
                        x1 = np.single(xc + FxRadius * np.cos((2 * np.pi * p) / XTNeighborPoints));
                        z1 = np.single(i + TInterval * np.sin((2 * np.pi * p) / XTNeighborPoints));

                        u = x1 - np.floor(x1-1);
                        v = z1 - np.floor(z1-1);
                        ltx = np.floor(x1-1);
                        lty = np.floor(z1-1);
                        lbx = np.floor(x1-1);
                        lby = np.ceil(z1-1);
                        rtx = np.ceil(x1-1);
                        rty = np.floor(z1-1);
                        rbx = np.ceil(x1-1);
                        rby = np.ceil(z1-1);
                        # the values of neighbors that do not fall exactly on
                        # pixels are estimated by bilinear interpolation of
                        # four corner points near to it.
                        CurrentVal = np.floor(VolData[int(yc), int(ltx), int(lty)] * (1 - u) * (1 - v) + VolData[int(yc), int(lbx), int(lby)] * (1 - u) * v + VolData[int(yc), int(rtx), int(rty)] * u * (1 - v) + VolData[int(yc), int(rbx), int(rby)] * u * v);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ^ FeaBin;

                        FeaBin = FeaBin + 1;

                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[1, BasicLBP + 1] = Histogram[1, BasicLBP + 1] + 1;
                    else:
                        Histogram[1, Code(BasicLBP + 1, 2) + 1] = Histogram[1, Code(BasicLBP + 1, 2) + 1] + 1;


                    ## In YT plane
                    BasicLBP = 0;
                    FeaBin = 0;
                    for p in range(0, YTNeighborPoints - 1):
                        # bilinear interpolation
                        y1 = np.single(yc - FyRadius * np.sin((2 * np.pi * p) / YTNeighborPoints));
                        z1 = np.single(i + TInterval * np.cos((2 * np.pi * p) / YTNeighborPoints));

                        u = y1 - np.floor(y1-1);
                        v = z1 - np.floor(z1-1);
                        ltx = np.floor(y1-1);
                        lty = np.floor(z1-1);
                        lbx = np.floor(y1-1);
                        lby = np.ceil(z1-1);
                        rtx = np.ceil(y1-1);
                        rty = np.floor(z1-1);
                        rbx = np.ceil(y1-1);
                        rby = np.ceil(z1-1);
                        # the values of neighbors that do not fall exactly on
                        # pixels are estimated by bilinear interpolation of
                        # four corner points near to it.
                        CurrentVal = np.floor(VolData[int(ltx), int(xc), int(lty)] * (1 - u) * (1 - v) + VolData[int(lbx), int(xc), int(lby)] * (1 - u) * v + VolData[int(rtx), int(xc), int(rty)] * u * (1 - v) + VolData[int(rbx), int(xc), int(rby)] * u * v);

                        if CurrentVal >= CenterVal:
                            BasicLBP = BasicLBP + 2 ^ FeaBin;

                        FeaBin = FeaBin + 1;

                    ## if Bincount is "0", it means basic LBP-TOP will be
                    ## computed and uniform patterns does not work in this case
                    ##. Otherwide it should be the number of the uniform
                    ##patterns, then "Code" keeps the lookup-table of the basic
                    ##LBP and uniform LBP
                    if Bincount == 0:
                        Histogram[2, BasicLBP] = Histogram[2, BasicLBP] + 1;
                    else:
                        Histogram[2, Code(BasicLBP + 1, 2)] = Histogram[2, Code(BasicLBP + 1, 2)] + 1;

                 ##
             ##
         ##


    ## normalization
    for j in range(3):
        Histogram[j, :] = Histogram[j, :]/np.sum(Histogram[j, :]);

    return Histogram