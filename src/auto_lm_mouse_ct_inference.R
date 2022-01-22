Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
Sys.setenv(CUDA_VISIBLE_DEVICES=2)
mygpu=Sys.getenv("CUDA_VISIBLE_DEVICES")
#
trnhfn = 'models/autopointsupdate_softmax_176_weights_3d_checkpoints5_GPU0_training_history.csv'
if ( file.exists( trnhfn ) ) {
  print("TRH")
  trnh = read.csv( trnhfn )
  plot( ts(trnh$loss ))
  points( trnh$testErr, col='red')
}

library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( reticulate )
library( ggplot2 )
library( tfdatasets )
library( glue )
mytype = "float32"

reoTemplate = antsImageRead( "templateImage.nii.gz" ) # antsImageClone( imgListTest[[templatenum]] )
ptTemplate = data.matrix( read.csv( "templatePoints.csv" ) )# ptListTest[[templatenum]]

fnsNew = Sys.glob("data/*[0-9].nii.gz")
fnsNewLM = Sys.glob("data/*[0-9]-LM.nii.gz")


# this is inference code
orinet =  createResNetModel3D(
       list(NULL,NULL,NULL,1),
       inputScalarsSize = 0,
       numberOfClassificationLabels = 6,
       layers = 1:4,
       residualBlockSchedule = c(3, 4, 6, 3),
       lowestResolution = 16,
       cardinality = 1,
       squeezeAndExcite = TRUE,
       mode = "regression")
load_model_weights_hdf5( orinet, "models/mouse_rotation_3D_GPU2.h5" )

unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1),
       numberOfOutputs = 55,
       numberOfLayers = 4,
       numberOfFiltersAtBaseLayer = 32,
       convolutionKernelSize = 3,
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = "nnUnetActivationStyle",
       mode = c("regression")
     )
findpoints = deepLandmarkRegressionWithHeatmaps( unet, activation='softmax', theta=NA )
#load_model_weights_hdf5( findpoints,   "models/autopointsfocused_sigmoid_128_weights_3d_checkpoints5_GPU2.h5" )
load_model_weights_hdf5( findpoints,   "models/autopointsupdate_softmax_176_weights_3d_checkpoints5_GPU0.h5" )
myaff = randomAffineImage( reoTemplate, "Rigid", sdAffine = 0 )[[2]]
idparams = getAntsrTransformParameters( myaff )
fixparams = getAntsrTransformFixedParameters( myaff )
templateCoM = getCenterOfMass( reoTemplate )

whichk = length( fnsNew ) # left out last subject as test
print( paste(  fnsNew[whichk], whichk ) )
locdim = dim( reoTemplate )
oimg = antsImageRead( fnsNew[whichk] ) %>% resampleImage( locdim, useVoxels=TRUE)
trulms = getCentroids( antsImageRead( fnsNewLM[whichk] ) )[,1:3]
img = histogramMatchImage( oimg, reoTemplate )
imgCoM = getCenterOfMass( iMath(img, "Normalize") )
imgarr = array( as.array( iMath(img, "Normalize") ), dim=c(1,locdim,1) )
print("deep rigid")
with(tf$device("/cpu:0"), {
      predRot <- predict( orinet, tf$cast( imgarr, mytype), batch_size = 1 )
    })
mm = matrix( predRot[1,], nrow=3, byrow=F)
mmm = cbind( mm, pracma::cross( mm[,1], mm[,2] ) )
mm = polarDecomposition( mmm )$Z
locparams = getAntsrTransformParameters( myaff )
locparams[1:9] = mm
locparams[10:length(locparams)] = (imgCoM - templateCoM )
setAntsrTransformParameters( myaff, locparams )
setAntsrTransformFixedParameters( myaff, templateCoM )
rotated = applyAntsrTransformToImage( myaff, img, reoTemplate )
# plot( reoTemplate, rotated, axis=3, nslices=21, ncolumns=7 )
print("classical rigid")
message("WE ARE STILL RUNNING ORINET BUT NOT USING IT AS ITS TRAINED ON DIFFERENT DATASPACE" )
qreg = antsRegistration( reoTemplate, img, "Rigid" ) # , initialTransform=myaff )
print("classical sim")
qreg = antsRegistration( reoTemplate, img, "Similarity", initialTransform=qreg$fwdtransforms )
print("classical aff")
qreg = antsRegistration( reoTemplate, img, "Affine", initialTransform=qreg$fwdtransforms )
img2LM = qreg$warpedmovout
print( antsImageMutualInformation( reoTemplate, img2LM) )
doviz=FALSE
if ( doviz )
  plot( reoTemplate, img2LM, axis=3, nslices=21, ncolumns=7 )
#################################################
img2LMcoords = coordinateImages( img2LM * 0 + 1 )
mycc = array( dim = c( 1, dim( img2LM ), 3 ) )
for ( jj in 1:3 ) mycc[1,,,,jj] = as.array( img2LMcoords[[jj]] )
imgarr[1,,,,1] = as.array( iMath( img2LM, "Normalize" ) )
telist = list(  tf$cast( imgarr, mytype), tf$cast( mycc, mytype) )
with(tf$device("/cpu:0"), {
      pointsoutte <- predict( findpoints, telist, batch_size = 1 )
      })
ptp = as.array(pointsoutte[[2]])[1,,]
ptimg = makePointsImage( ptp, img2LM*0+1, radius=0.2 )  %>% iMath("GD",3)
print(sort(unique(ptimg)))
if ( doviz)
  plot( img2LM, ptimg, nslices = 21, ncolumns = 7, axis=3 )
ptmask = thresholdImage( ptimg, 1, 2 )
ptmaskdil = iMath( ptmask, "MD", 8 )
ptpb = antsApplyTransformsToPoints( 3, ptp, qreg$fwdtransforms, whichtoinvert=c(FALSE) )
# roughly percent errror
print( norm( (trulms - data.matrix(ptpb) ))/norm(trulms) )
ptimg2 = makePointsImage( ptpb, img*0+1, radius=0.2 ) %>% iMath("GD",2)
# plot( oimg, ptimg2, nslices = 21, ncolumns = 7, axis=3 )
antsImageWrite( ptimg2, '/tmp/temp.nii.gz' )
