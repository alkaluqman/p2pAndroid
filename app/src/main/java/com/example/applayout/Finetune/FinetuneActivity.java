package com.example.applayout.Finetune;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;


import com.example.applayout.BaseActivity;
import com.example.applayout.MainActivity;
import com.example.applayout.R;
import com.example.applayout.Report.ReportActivity2;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.app.AlertDialog;
import android.view.LayoutInflater;
import android.widget.EditText;
import android.widget.Toast;

import com.example.applayout.Metrics.MetricTracking;

public class FinetuneActivity extends BaseActivity {
    Context context = this;
    ProgressBar progressBar;
    TextView status, text, parameters;

    Button btStartFinetune;
    Button btReport;
    final boolean THREAD_TEST = false;

    private String model;
    private int numEpochs;
    private int batchSize;
    private int imgHeight;
    private int imgWidth;
    private int numTrainings;

    private class TrainModelTask extends AsyncTask<Void, Integer, Void> {

        private String model;
        private int numEpochs;
        private int batchSize;
        private int imgHeight;
        private int imgWidth;
        private int numTrainings;

        public TrainModelTask(String model, int numEpochs, int batchSize, int imgHeight, int imgWidth, int numTrainings) {
            this.model = model;
            this.numEpochs = numEpochs;
            this.batchSize = batchSize;
            this.imgHeight = imgHeight;
            this.imgWidth = imgWidth;
            this.numTrainings = numTrainings;
        }

        protected void onPreExecute() {
            super.onPreExecute();
//            progressBar = findViewById(R.id.progressBar);
//            progressBar.setVisibility(View.VISIBLE);

            // Ensure UI update happens on the main thread
            runOnUiThread(() -> {
                progressBar = findViewById(R.id.progressBar);
                progressBar.setVisibility(View.VISIBLE);
            });
        }

        protected Void doInBackground(Void... voids) {
            try (Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(), model))) {
//                int NUM_EPOCHS = 100;
////                int NUM_EPOCHS = 100;
//                int BATCH_SIZE = 100;
//                int IMG_HEIGHT = 28;
//                int IMG_WIDTH = 28;
//                int NUM_TRAININGS = 60000;
//                int NUM_BATCHES = NUM_TRAININGS / BATCH_SIZE;

                //List<FloatBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
                //List<FloatBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);
                List<FloatBuffer> trainImageBatches = new ArrayList<>(10);
                List<FloatBuffer> trainLabelBatches = new ArrayList<>(10);

                for (int i = 0; i < 10; ++i) {
                    String imagePath = "test_images/image" + i + ".png";
                    String labelPath = "labels/label" + i + ".txt";
                    FloatBuffer trainImages = readImageAsFloatBuffer(context, imagePath, imgWidth, imgHeight);
                    FloatBuffer trainLabels = readLabelAsFloatBuffer(context, labelPath, 10); // Assuming 10 classes

                    if (trainImages != null && trainLabels != null) {
                        trainImageBatches.add(trainImages);
                        trainLabelBatches.add(trainLabels);
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
                    for (int batchIdx = 0; batchIdx < 10; ++batchIdx) {
                        Map<String, Object> inputs = new HashMap<>();
                        inputs.put("x", trainImageBatches.get(batchIdx));
                        inputs.put("y", trainLabelBatches.get(batchIdx));

                        Map<String, Object> outputs = new HashMap<>();
                        FloatBuffer lossBuffer = FloatBuffer.allocate(1);
                        outputs.put("loss", lossBuffer);

                        anotherInterpreter.runSignature(inputs, outputs, "train");
                        final int progressPercentage = (epoch * 100) / numEpochs;
                        progressBar.setProgress(progressPercentage);
                        //float lossValue = lossBuffer.get(0);

                        // Record the last loss.
                        if (batchIdx == 10 - 1) losses[epoch] = lossBuffer.get(0);
                    }

                    // Print the loss output for every 10 epochs.
                    String message = "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch];
                    if (THREAD_TEST) {
                        message += " in Thread ID: " + Thread.currentThread().getId();
                    }
                    if ((epoch + 1) % 10 == 0) {
                        System.out.println(message);
                    }
                }
                saveModelWeights(anotherInterpreter);
            } catch (IOException e) {
                Log.e("ReportActivity", "Error", e);
            }
            return null;
        }

        protected void onProgressUpdate(Integer... progress) {
            super.onProgressUpdate(progress);

            // Ensure UI update happens on the main thread
            runOnUiThread(() -> {
                progressBar.setProgress(progress[0]);
            });
        }

        protected void onPostExecute(Void result) {
            super.onPostExecute(result);

            // Ensure UI update happens on the main thread
            runOnUiThread(() -> {
                progressBar.setVisibility(View.GONE);
                text.setText("Training Completed");
                btReport.setVisibility(View.VISIBLE);
            });
        }
    }

    protected void onCreate(Bundle savedInstanceState) {
        setContentView(R.layout.activity_finetune);
        super.onCreate(savedInstanceState);
        //OnDeviceTraining.onDeviceTraining();
        text = findViewById(R.id.finetune_text);
        status = findViewById(R.id.finetune_status);
        btReport = findViewById(R.id.btReport);
        status.setText("Learning From: OPPO R11\nML Objective: " + MainActivity.string + "\nML Model Size: 0.82MB");
        btReport.setVisibility(View.GONE);

        btReport.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), ReportActivity2.class);
                startActivity(intent);
            }
        });

        Button btOpenDialog = findViewById(R.id.btOpenDialog);
        btOpenDialog.setOnClickListener(v -> showNumberInputDialog());

        parameters = findViewById(R.id.finetune_parameters);
        parameters.setText("No parameters set. Please enter finetune parameters.");

        btStartFinetune = findViewById(R.id.btStartFinetune);
        btStartFinetune.setOnClickListener(v -> {
                    new Thread(() -> {
                        HashMap<String, Object> trackingResults = MetricTracking.doWithTracking(FinetuneActivity.this::finetune);
                        // Log results (replace this with saving to a file or another desired action)
                        System.out.println("Tracking Results: " + trackingResults);
                    }
                    ).start();
                }
        );

        if (THREAD_TEST) {
            final int NUM_MODELS = 10;

            // EXPERIMENT USING THREADS
            ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
            System.out.println("Size: " + Runtime.getRuntime().availableProcessors());

            for (int i = 0; i < NUM_MODELS; i++) {
                executor.execute(() -> {
                    new TrainModelTask("model.tflite", 100, 100, 28, 28, 60000).execute();
                });
            }

            executor.shutdown(); // Ensures all tasks finish
        } else {
            new TrainModelTask("model.tflite", 100, 100, 28, 28, 60000).execute();
        }

        /*context = this;

        try (Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(),"model.tflite"))) {
            int NUM_EPOCHS = 100;
            int BATCH_SIZE = 100;
            int IMG_HEIGHT = 28;
            int IMG_WIDTH = 28;
            int NUM_TRAININGS = 60000;
            int NUM_BATCHES = NUM_TRAININGS / BATCH_SIZE;

            List<FloatBuffer> trainImageBatches = new ArrayList<>(NUM_BATCHES);
            List<FloatBuffer> trainLabelBatches = new ArrayList<>(NUM_BATCHES);

            // Prepare training batches.
            for (int i = 0; i < NUM_BATCHES; ++i) {
                ByteBuffer trainImageBuffer = ByteBuffer.allocateDirect(4 * IMG_HEIGHT * IMG_WIDTH).order(ByteOrder.nativeOrder());
                FloatBuffer trainImages = trainImageBuffer.asFloatBuffer();

                ByteBuffer trainLabelsBuffer = ByteBuffer.allocateDirect(4 * 10).order(ByteOrder.nativeOrder());
                FloatBuffer trainLabels = trainLabelsBuffer.asFloatBuffer();

                // Fill the data values...
                trainImageBatches.add((FloatBuffer) trainImages.rewind());
                trainLabelBatches.add((FloatBuffer) trainLabels.rewind());
            }

            // Run training for a few steps.
            float[] losses = new float[NUM_EPOCHS];
            for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
                for (int batchIdx = 0; batchIdx < NUM_BATCHES; ++batchIdx) {
                    Map<String, Object> inputs = new HashMap<>();
                    inputs.put("x", trainImageBatches.get(batchIdx));
                    inputs.put("y", trainLabelBatches.get(batchIdx));

                    Map<String, Object> outputs = new HashMap<>();
                    FloatBuffer lossBuffer = FloatBuffer.allocate(1);
                    outputs.put("loss", lossBuffer);

                    anotherInterpreter.runSignature(inputs, outputs, "train");

                    //float lossValue = lossBuffer.get(0);

                    // Record the last loss.
                    if (batchIdx == NUM_BATCHES - 1) losses[epoch] = lossBuffer.get(0);
                }

                // Print the loss output for every 10 epochs.
                if ((epoch + 1) % 10 == 0) {
                    System.out.println(
                            "Finished " + (epoch + 1) + " epochs, current loss: " + losses[epoch]);
                }
            }
        } catch (IOException e){
            e.printStackTrace();
        }*/
    }

    private void showNumberInputDialog() {
        // Inflate custom layout for the dialog
        LayoutInflater inflater = LayoutInflater.from(this);
        View dialogView = inflater.inflate(R.layout.dialog_number_input, null);

        EditText inputNumEpochs = dialogView.findViewById(R.id.inputNumEpochs);
        EditText inputBatchSize = dialogView.findViewById(R.id.inputBatchSize);
        EditText inputImgHeight = dialogView.findViewById(R.id.inputImgHeight);
        EditText inputImgWidth = dialogView.findViewById(R.id.inputImgWidth);
        EditText inputNumTrainings = dialogView.findViewById(R.id.inputNumTrainings);

        // Pre-fill with default values
        inputNumEpochs.setText(this.numEpochs == 0 ? "100" : Integer.toString(this.numEpochs));
        inputBatchSize.setText(this.batchSize == 0 ? "100" : Integer.toString(this.batchSize));
        inputImgHeight.setText(this.imgHeight == 0 ? "28" : Integer.toString(this.imgHeight));
        inputImgWidth.setText(this.imgWidth == 0 ? "28" : Integer.toString(this.imgWidth));
        inputNumTrainings.setText(this.numTrainings == 0 ? "60000" : Integer.toString(this.numTrainings));

        // Build the AlertDialog
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter Finetune Parameters")
                .setView(dialogView)
                .setPositiveButton("OK", (dialog, which) -> {
                    String numEpochs = inputNumEpochs.getText().toString().trim();
                    String batchSize = inputBatchSize.getText().toString().trim();
                    String imgHeight = inputImgHeight.getText().toString().trim();
                    String imgWidth = inputImgWidth.getText().toString().trim();
                    String numTrainings = inputNumTrainings.getText().toString().trim();

                    if (numEpochs.isBlank() || batchSize.isBlank() || imgHeight.isBlank() || imgWidth.isBlank() || numTrainings.isBlank()) {
                        Toast.makeText(this, "One or more fields are empty!", Toast.LENGTH_SHORT).show();
                    } else {
                        parseAndSetFinetuneParameters(numEpochs, batchSize, imgHeight, imgWidth, numTrainings);
                        Toast.makeText(this, "Input Saved", Toast.LENGTH_SHORT).show();
                    }

                })
                .setNegativeButton("Cancel", (dialog, which) -> dialog.dismiss());

        builder.create().show();
    }

    private boolean hasMissingParams() {
        if (this.numEpochs == 0 || this.batchSize == 0 || this.imgHeight == 0 || this.imgWidth == 0 || this.numTrainings == 0) {
            runOnUiThread(() -> {
                Toast.makeText(getApplicationContext(), "No parameters set!", Toast.LENGTH_SHORT).show();
            });
            return true;
        }
        return false;
    }

    /**
     * This is a blocking method as .get() is required to ensure knowledge of the doInBackground() method's completion.
     * To avoid blocking the UI thread, ensure that this function's caller is a separate Thread.
     */
    private void finetune() {
        if (hasMissingParams()) return;

        try {
            new TrainModelTask("model.tflite", this.numEpochs, this.batchSize, 28, this.imgWidth, this.numTrainings).execute().get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    private void parseAndSetFinetuneParameters(String inputNumEpochs, String inputBatchSize, String inputImgHeight, String inputImgWidth, String inputNumTrainings) {
        try {
            int numEpochs = Integer.parseInt(inputNumEpochs);
            int batchSize = Integer.parseInt(inputBatchSize);
            int imgHeight = Integer.parseInt(inputImgHeight);
            int imgWidth = Integer.parseInt(inputImgWidth);
            int numTrainings = Integer.parseInt(inputNumTrainings);

            this.numEpochs = numEpochs;
            this.batchSize = batchSize;
            this.imgHeight = imgHeight;
            this.imgWidth = imgWidth;
            this.numTrainings = numTrainings;

            String params = "Number of Epochs: " + numEpochs + " \n Batch Size: " + batchSize + "\n Image Height: " + imgHeight + "\n Image Width: " + imgWidth + "\n Number of Trainings: " + numTrainings;

            parameters.setText(params);

        } catch (NumberFormatException e) {
            Toast.makeText(getApplicationContext(), "Error: Please enter a valid number!", Toast.LENGTH_SHORT).show();
        }
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public FloatBuffer readImageAsFloatBuffer(Context context, String assetPath, int imgWidth, int imgHeight) {
        try {
            InputStream is = context.getAssets().open(assetPath);
            Bitmap bitmap = BitmapFactory.decodeStream(is);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imgHeight * imgWidth).order(ByteOrder.nativeOrder());
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();

            int[] pixels = new int[imgWidth * imgHeight];
            resizedBitmap.getPixels(pixels, 0, imgWidth, 0, 0, imgWidth, imgHeight);

            for (int pixel : pixels) {
                float red = ((pixel >> 16) & 0xFF) / 255.0f;
                float green = ((pixel >> 8) & 0xFF) / 255.0f;
                float blue = (pixel & 0xFF) / 255.0f;
                floatBuffer.put(red);  // Just an example; adjust as needed for your model
            }
            floatBuffer.rewind();
            return floatBuffer;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public FloatBuffer readLabelAsFloatBuffer(Context context, String assetPath, int numClasses) {
        try {
            InputStream is = context.getAssets().open(assetPath);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            int labelIndex = Integer.parseInt(reader.readLine().trim());

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * numClasses).order(ByteOrder.nativeOrder());
            FloatBuffer labelBuffer = byteBuffer.asFloatBuffer();
            labelBuffer.put(labelIndex, 1.0f);  // One-hot encoding
            labelBuffer.rewind();
            return labelBuffer;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private void saveModelWeights(Interpreter interpreter) {
        String filename = "trained_model_weights.ckpt";
        File file = new File(context.getFilesDir(), filename);
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", file.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();

        interpreter.runSignature(inputs, outputs, "save");
        Log.d("FinetuneActivity", "Model weights saved to " + file.getAbsolutePath());
    }
}
