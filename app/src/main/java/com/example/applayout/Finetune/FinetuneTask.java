package com.example.applayout.Finetune;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
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

    private final Context context;
    private final String modelFileName;
    private final String datasetDirName;
    private final File filesDir;
    private final int numEpochs;
    private final int batchSize;

    private final String newModelFileName;

    public FinetuneTask(Context context, File filesDir, String modelFileName, String datasetDirName, int numEpochs, int batchSize, String newModelFileName) {
        this.context = context.getApplicationContext();
        this.filesDir = filesDir;
        this.modelFileName = modelFileName;
        this.datasetDirName = datasetDirName;
        this.numEpochs = numEpochs;
        this.batchSize = batchSize;
        this.newModelFileName = newModelFileName;
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


        finetuneManual(context, filesDir, modelFileName, datasetDirName, numEpochs, batchSize, newModelFileName);
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

    private void finetuneManual(Context context, File filesDir, String modelFileName, String datasetDirName, int numEpochs, int batchSize, String newModelFileName) {
        Log.d(this.getClass().getName(), "Beginning finetuneManual...");
        final String MODEL_EXT = ".ckpt";

        try {
            Interpreter anotherInterpreter = new Interpreter(FinetuneUtils.loadModelFile(context.getAssets(), "model.tflite")); // default base model
            String modelFileAbsolutePath = FinetuneUtils.getAbsolutePathFromFilesDir(filesDir, "models", modelFileName + MODEL_EXT);
            String datasetDirAbsolutePath = FinetuneUtils.getAbsolutePathFromFilesDir(filesDir, "datasets", datasetDirName);

            // Load weights from checkpoint file (this is where the checkpoint path is used)
            if (!modelFileAbsolutePath.isEmpty()) {
                FinetuneUtils.restoreWeightsFromCheckpoint(anotherInterpreter, modelFileAbsolutePath);
            }

            // Get image and label data
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

            int[] modelInputShape = FinetuneUtils.getModelInputShape(context.getAssets(), "model.tflite");
            if (modelInputShape == null)
                throw new RuntimeException("Failed to get model shape for model: " + modelFileName + MODEL_EXT);
            Log.d("FinetuneActivity", "Model Input Shape: " + Arrays.toString(modelInputShape));
            int imgWidth = modelInputShape[1];
            int imgHeight = modelInputShape[2];

            List<FloatBuffer> trainImageBatches = new ArrayList<>(numImages);
            List<FloatBuffer> trainLabelBatches = new ArrayList<>(numImages);

            // Process images and labels for training
            for (int i = 0; i < numImages; ++i) {
                String imageFileName = imageFileNames.get(i);
                Integer labelIndex = labelMap.get(imageFileName);

                if (labelIndex == null)
                    throw new RuntimeException("No label found for image: " + imageFileName + " in dataset: " + datasetDirName);

                FloatBuffer trainImage = FinetuneUtils.readImageAsFloatBuffer(datasetDirAbsolutePath, imageFileName, imgWidth, imgHeight);
                FloatBuffer trainLabel = FinetuneUtils.readLabelAsFloatBuffer(labelIndex, 10);

                if (trainImage != null && trainLabel != null) {
                    trainImageBatches.add(trainImage);
                    trainLabelBatches.add(trainLabel);
                } else {
                    Log.e("FinetuneActivity", "Failed to read image or label for batch " + i);
                }
            }

            // Train model for the given number of epochs
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

                    // Record the last loss.
                    if (batchIdx == batchSize - 1) losses[epoch] = lossBuffer.get(0);
                }

                // Print the loss output for every 10 epochs.
                String message = "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch];
                if ((epoch + 1) % 10 == 0) {
                    System.out.println(message);
                }
            }

            FinetuneUtils.saveModelWeights(anotherInterpreter, filesDir, newModelFileName);
            FinetuneUtils.saveLosses(filesDir, losses);
            Log.d(this.getClass().getName(), "Completed finetuneManual!");
        } catch (IOException e) {
            Log.e(this.getClass().getName(), "Error", e);
        }
    }

}
