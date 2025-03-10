package com.example.applayout.Finetune;

import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An instance of this class represents an instance of Finetuning.
 */
public class FinetuneTask extends AsyncTask<Void, Integer, Void> {

    private final String modelFileAbsolutePath;
    private final String datasetDirAbsolutePath;
    private final int numEpochs;
    private final int batchSize;

    public FinetuneTask(String modelFileAbsolutePath, String datasetDirAbsolutePath, int numEpochs, int batchSize) {
        this.modelFileAbsolutePath = modelFileAbsolutePath;
        this.datasetDirAbsolutePath = datasetDirAbsolutePath;
        this.numEpochs = numEpochs;
        this.batchSize = batchSize;
    }

    protected void onPreExecute() {
        super.onPreExecute();
//        // Ensure UI update happens on the main thread
//        runOnUiThread(() -> {
//            progressBar = findViewById(R.id.progressBar);
//            progressBar.setVisibility(View.VISIBLE);
//        });
    }

    protected Void doInBackground(Void... voids) {
        finetuneManual(this.modelFileAbsolutePath, this.datasetDirAbsolutePath, numEpochs, batchSize);
        return null;
    }

    protected void onProgressUpdate(Integer... progress) {
        super.onProgressUpdate(progress);

        // Ensure UI update happens on the main thread
//        runOnUiThread(() -> {
//            progressBar.setProgress(progress[0]);
//        });
    }

    protected void onPostExecute(Void result) {
        super.onPostExecute(result);

        // Ensure UI update happens on the main thread
//        runOnUiThread(() -> {
//            progressBar.setVisibility(View.GONE);
//            text.setText("Training Completed");
//            btReport.setVisibility(View.VISIBLE);
//        });
    }

    private void finetuneManual(String modelFileAbsolutePath, String datasetDirAbsolutePath, int numEpochs, int batchSize) {
        Log.d(this.getClass().getName(), "Beginning finetuneManual...");
        try (Interpreter anotherInterpreter = new Interpreter(FinetuneUtils.loadModelFile(modelFileAbsolutePath))) {
//            List<FloatBuffer> trainImageBatches = new ArrayList<>(10);
//            List<FloatBuffer> trainLabelBatches = new ArrayList<>(10);
//
//            // TODO: Update with dataset
//            for (int i = 0; i < 10; ++i) {
//                String imagePath = "test_images/image" + i + ".png";
//                String labelPath = "labels/label" + i + ".txt";
//                FloatBuffer trainImages = FinetuneUtils.readImageAsFloatBuffer(context, imagePath, imgWidth, imgHeight);
//                FloatBuffer trainLabels = FinetuneUtils.readLabelAsFloatBuffer(context, labelPath, 10); // Assuming 10 classes
//            }

            List<String> imageFileNames = FinetuneUtils.getImageFileNamesFromDataset(datasetDirAbsolutePath); // relative file names
            HashMap<String, Integer> labelMap = FinetuneUtils.getLabelMapFromDataset(datasetDirAbsolutePath);
            Log.d(this.getClass().getName(), "imageFileNames: " + imageFileNames);
            assert labelMap != null;
            Log.d(this.getClass().getName(), "labelMap: " + labelMap);
            int numImages = imageFileNames.size();
            Log.d(this.getClass().getName(), "numImages: " + numImages);

            int numClasses = FinetuneUtils.getNumClasses(labelMap);
            Log.d(this.getClass().getName(), "Number of classes: " + numClasses);

            HashMap<Integer, Integer> originalLabelsToDatasetLabelsMap = FinetuneUtils.getOriginalLabelsToDatasetLabelsMap(labelMap);
            Log.d(this.getClass().getName(), "originalLabelsToDatasetLabelsMap : " + originalLabelsToDatasetLabelsMap);

            int[] modelInputShape = FinetuneUtils.getModelInputShape(modelFileAbsolutePath);
            if (modelInputShape == null)
                throw new RuntimeException("Failed to get model shape for model: " + modelFileAbsolutePath);
            Log.d("FinetuneActivity", "Model Input Shape: " + Arrays.toString(modelInputShape));
            int imgWidth = modelInputShape[1];
            int imgHeight = modelInputShape[2];

            List<FloatBuffer> trainImageBatches = new ArrayList<>(numImages);
            List<FloatBuffer> trainLabelBatches = new ArrayList<>(numImages);

            for (int i = 0; i < numImages; ++i) {
                String imageFileName = imageFileNames.get(i);
                Integer labelIndex = labelMap.get(imageFileName);

                if (labelIndex == null)
                    throw new RuntimeException("No label found for image: " + imageFileName + " in dataset: " + datasetDirAbsolutePath);

//                int datasetLabelIndex = originalLabelsToDatasetLabelsMap.get(labelIndex);

                FloatBuffer trainImage = FinetuneUtils.readImageAsFloatBuffer(datasetDirAbsolutePath, imageFileName, imgWidth, imgHeight);
                FloatBuffer trainLabel = FinetuneUtils.readLabelAsFloatBuffer(labelIndex, 10);

                if (trainImage != null && trainLabel != null) {
                    trainImageBatches.add(trainImage);
                    trainLabelBatches.add(trainLabel);
                } else {
                    Log.e("FinetuneActivity", "Failed to read image or label for batch " + i);
                }
            }

            // Prepare training batches.
                /*for (int i = 0; i < NUM_BATCHES; ++i) {
                    ByteBuffer trainImageBuffer = ByteBuffer.allocateDirect(4 * IMG_HEIGHT * IMG_WIDTH).order(ByteOrder.nativeOrder());
                    FloatBuffer trainImages = trainImageBuffer.asFloatBuffer();

                    ByteBuffer trainLabelsBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                    FloatBuffer trainLabels = trainLabelsBuffer.asFloatBuffer();

                    // Fill the data values...
                    trainImageBatches.add((FloatBuffer) trainImages.rewind());
                    trainLabelBatches.add((FloatBuffer) trainLabels.rewind());
                }*/

            // Run training for a few steps.
            float[] losses = new float[numEpochs];
            for (int epoch = 0; epoch < numEpochs; ++epoch) {
                for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", trainImageBatches.get(batchIdx));
                    inputs.put("y", trainLabelBatches.get(batchIdx));

                    Map<String, Object> outputs = new HashMap<>();
                    FloatBuffer lossBuffer = FloatBuffer.allocate(1);
                    outputs.put("loss", lossBuffer);

                    anotherInterpreter.runSignature(inputs, outputs, "train");
                    final int progressPercentage = (epoch * 100) / numEpochs;
//                    progressBar.setProgress(progressPercentage);
                    //float lossValue = lossBuffer.get(0);

                    // Record the last loss.
                    if (batchIdx == batchSize - 1) losses[epoch] = lossBuffer.get(0);
                }

                // Print the loss output for every 10 epochs.
                String message = "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch];
                if ((epoch + 1) % 10 == 0) {
                    System.out.println(message);
                }
            }
//            FinetuneUtils.saveModelWeights(context, anotherInterpreter);
            Log.d(this.getClass().getName(), "Completed finetuneManual!");
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Error", e);
        }
    }
}
