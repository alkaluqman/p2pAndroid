package com.example.applayout.Report;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.applayout.Communications.CommunicationActivity;
import com.example.applayout.R;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ReportActivity2 extends AppCompatActivity {
    TextView modelDetails;
    ListView lvReport;
    private Context context;
    Button btStart;
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_report);

        modelDetails = findViewById(R.id.modelDetails);
        modelDetails.setText("Model Name: model.tflite\nLast Trained On: 23/04/2024\n");

        lvReport = findViewById(R.id.lvReport);
        ArrayList<String> reportList = new ArrayList<>();
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, reportList);
        lvReport.setAdapter(adapter);

        context = this;

        FloatBuffer output = null;

        try (Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(),"model.tflite"))) {
            restoreModelWeights(anotherInterpreter);

            FloatBuffer testImages = loadTestData();
            if (testImages == null) {
                Log.e("ReportActivity", "Failed to load test images.");
                return; // Exit if no data could be loaded
            }

            int[] trueLabels = loadTrueLabels();
            int correctPredictions = 0;

            for (int i = 0; i < 10; i++) {
                FloatBuffer singleImage = getSingleImage(testImages, i); // Extract one test image
                Map<String, Object> inputs = new HashMap<>();
                Map<String, Object> outputs = new HashMap<>();
                output = FloatBuffer.allocate(10); // Assume the model outputs 10 classes

                inputs.put("x", singleImage); // Use the extracted single image as input
                outputs.put("output", output);

                anotherInterpreter.runSignature(inputs, outputs, "infer");
                output.rewind(); // Ensure the output buffer is ready to be read

                int predictedLabel = argMax(output.array());
                float confidence = output.get(predictedLabel);

                if (predictedLabel == trueLabels[i]) {
                    correctPredictions++;
                }

                String resultText = String.format("Image %d: Predicted Label = %d, Confidence = %.2f%%", i, predictedLabel, confidence * 100);
                reportList.add(resultText);
            }

            float accuracy = (float) correctPredictions / 10 ;
            modelDetails.setText("Model Name: model.tflite\nLast Trained On: 23/04/2024\nModel Accuracy: " + (accuracy * 100) + "%");
            /*
            //ByteBuffer testImageBuffer = ByteBuffer.allocateDirect(4 * 28 * 28).order(ByteOrder.nativeOrder());
            //FloatBuffer testImages = testImageBuffer.asFloatBuffer();

            ByteBuffer testLabelsBuffer = ByteBuffer.allocateDirect(4 * 10 * 4).order(ByteOrder.nativeOrder());
            output = testLabelsBuffer.asFloatBuffer();

            // Fill the test data.

            // Run the inference.
            //Map<String, Object> inputs = new HashMap<>();
            //inputs.put("x", testImages.rewind());
            //Map<String, Object> outputs = new HashMap<>();
            //outputs.put("output", output);
            //anotherInterpreter.runSignature(inputs, outputs, "infer");
            //output.rewind();

            for (int i = 0; i < trueLabels.length; i++) {
                Map<String, Object> inputs = new HashMap<>();
                Map<String, Object> outputs = new HashMap<>();
                inputs.put("x", testImages.rewind());
                outputs.put("output", output);
                inputs.put("x", testImages.get(i)); // Adjust this line according to how your data is structured

                // Run the TensorFlow Lite model
                output.clear(); // Reset position and limit if it was previously read
                anotherInterpreter.runSignature(inputs, outputs, "infer");
                output.rewind(); // Reset position to read from the beginning

                if (output.limit() == 0) {
                    throw new RuntimeException("Output buffer is empty, inference failed or output buffer not set correctly.");
                }

                int predictedLabel = argMax(output.array()); // Find the index of the maximum confidence score
                if (predictedLabel == trueLabels[i]) {
                    correctPredictions++;
                }
            }

            float accuracy = (float) correctPredictions / trueLabels.length;
            modelDetails.setText("Model Name: model.tflite\nLast Trained On: 20/03/2024\nModel Accuracy: " + (accuracy*100) + "%");

            // Process the result to get the final category values. (working)
            /*int[] testLabels = new int[NUM_TESTS];
            float[] confidenceScore = new float[NUM_TESTS];
            for (int i = 0; i < NUM_TESTS; ++i) {
                int maxIndex = 0; // Start assuming the first index has the maximum value
                float maxValue = output.get(i * 10); // Assume first element is the largest initially
                for (int j = 1; j < 10; ++j) {
                    float currentValue = output.get(i * 10 + j);
                    if (currentValue > maxValue) {
                        maxValue = currentValue;
                        maxIndex = j;
                    }
                }
                testLabels[i] = maxIndex;
                confidenceScore[i] = maxValue;

                reportList.add("Test " + i + ": Label = " + maxIndex + ", Confidence = " + String.format("%.2f", maxValue));

                //Log.d("ReportActivity", "Test " + i + ": Label = " + maxIndex + ", Confidence = " + maxValue);
            }*/
        } catch (IOException e) {
            e.printStackTrace();
        } catch (IndexOutOfBoundsException ex){
            if (output != null) {
                Log.e("ReportActivity", "Error accessing buffer at position: " + output.position() + " with limit: " + output.limit(), ex);
            }
            throw ex;
        } catch (BufferUnderflowException ez) {
            Log.e("ReportActivity", "Not enough data in buffer to process image", ez);
        }

        /*Model.Options options;
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            options = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
        } else {
            options = new Model.Options.Builder().setNumThreads(4).build();
        }
            //Export the trained weights as checkpoint file
            File outputFile = new File(getFilesDir(), "checkpoint.ckpt");
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("checkpoint_path", outputFile.getAbsolutePath());
            Map<String, Object> outputs = new HashMap<>();
            anotherInterpreter.runSignature(inputs, outputs, "save");
            }

            //model inference
            ByteBuffer inputBuffer = prepareInputData();
            FloatBuffer outputBuffer = FloatBuffer.allocate(360000);
            Map<Integer, Object> outputs = new HashMap<>();
            int outputTensorIndex = 0;
            outputs.put(outputTensorIndex, outputBuffer);
            anotherInterpreter.runForMultipleInputsOutputs(new Object[]{inputBuffer}, outputs);
            float[] results = new float[4];
            outputBuffer.get(results);
            Log.e("Output: ", Arrays.toString(outputBuffer.array()));
            anotherInterpreter.close();
        } catch (IOException e){
            e.printStackTrace();
        }*/

        btStart = findViewById(R.id.btStartFL);
        btStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(view.getContext(), CommunicationActivity.class);
                startActivity(intent);
            }
        });
    }

    private int argMax(float[] array) {
        int best = -1;
        float bestConfidence = 0.0f;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > bestConfidence) {
                bestConfidence = array[i];
                best = i;
            }
        }
        return best;
    }

    private ByteBuffer prepareInputData() {
        int batchSize = 1;
        int inputHeight = 300;
        int inputWidth = 300;
        int channelCount = 3;

        int inputSize = batchSize * inputHeight * inputWidth * channelCount;

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputSize);
        inputBuffer.order(ByteOrder.nativeOrder());

        return inputBuffer;
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

    private FloatBuffer loadTestData() throws IOException {
        AssetManager assetManager = getAssets();
        String[] files = assetManager.list("test_images");
        if (files == null || files.length == 0) {
            Log.e("loadTestData", "No image files found");
            return null; // Early return or throw exception
        }
        // Assuming the model expects a single channel (grayscale), 28x28 input images
        ByteBuffer imageData = ByteBuffer.allocateDirect(files.length * 28 * 28 * 4); // 4 bytes per float
        imageData.order(ByteOrder.nativeOrder());
        for (String file : files) {
            try (InputStream is = assetManager.open("test_images/" + file)) {
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                bitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, false);
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int pixel = bitmap.getPixel(j, i);
                        float normalized = (pixel & 0xFF) / 255.0f;
                        imageData.putFloat(normalized);
                    }
                }
            } catch (IOException e) {
                Log.e("loadTestData", "Failed to load or process file: " + file, e);
                return null;
            }
        }
        imageData.rewind();
        return imageData.asFloatBuffer();
    }



    private int[] loadTrueLabels() throws IOException{
        ArrayList<Integer> labelList = new ArrayList<>();
        try(InputStream is = getAssets().open("labels.txt")){
            BufferedReader read = new BufferedReader(new InputStreamReader(is));
            String line;
            while ((line = read.readLine()) != null){
                labelList.add(Integer.parseInt(line.trim()));
            }
        }
        int[] labels = new int[labelList.size()];
        for(int i=0; i<labelList.size(); i++){
            labels[i] = labelList.get(i);
        }
        return labels;
    }

    private FloatBuffer getSingleImage(FloatBuffer buffer, int index) {
        int imageSize = 28 * 28; // Total number of floats per image
        int startPosition = index * imageSize;

        if (buffer.limit() < startPosition + imageSize) {
            Log.e("ReportActivity", "Insufficient data in buffer to read image at index " + index);
            throw new BufferUnderflowException();
        }

        // Allocate a new buffer for the single image
        FloatBuffer singleImage = FloatBuffer.allocate(imageSize);

        // Save the current position of the buffer
        int oldPosition = buffer.position();

        // Read data from the main buffer into the image buffer
        buffer.position(startPosition);
        float[] temp = new float[imageSize];
        buffer.get(temp, 0, imageSize);  // Safely read data into the temporary array
        singleImage.put(temp);  // Transfer data from the array to the FloatBuffer

        // Restore the original position of the main buffer
        buffer.position(oldPosition);

        singleImage.rewind();  // Prepare the single image buffer for reading
        return singleImage;
    }

    private void restoreModelWeights(Interpreter interpreter) {
        String filename = "trained_model_weights.ckpt";
        File file = new File(context.getFilesDir(), filename);
        if (!file.exists()) {
            Log.d("FederatedLearning", "No saved model weights to load.");
            return;
        }

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", file.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();

        interpreter.runSignature(inputs, outputs, "restore");
        Log.d("FederatedLearning", "Model weights restored from " + file.getAbsolutePath());
    }
}
