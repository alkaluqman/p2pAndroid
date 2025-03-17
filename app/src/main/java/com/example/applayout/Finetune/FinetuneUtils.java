package com.example.applayout.Finetune;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class FinetuneUtils {

    public static void saveLosses(File filesDir, float[] losses) throws IOException {
        File file = new File(filesDir, "last_run_losses");
        try (FileWriter writer = new FileWriter(file)) {
            writer.write("epoch,loss\n");
            for (int epoch = 0; epoch < losses.length; epoch++) {
                writer.write((epoch + 1) * 10 + "," + losses[epoch] + "\n");
            }
            Log.d("FinetuneUtils", "Loss data saved to: " + file.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void getBaseCkptFile(Context context, File filesDir, String modelFileName) {
        final String contextName = "FinetuneUtils";
        Log.d(contextName, "Spawning base .ckpt file...");

        try {
            Interpreter anotherInterpreter = new Interpreter(loadModelFile(context.getAssets(), "model.tflite")); // default base model

            saveModelWeights(anotherInterpreter, filesDir, modelFileName);
            Log.d(contextName, "Base .ckpt file spawned!");
        } catch (IOException e) {
            Log.e(contextName, "Error", e);
        }
    }

    public static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static String getAbsolutePathFromFilesDir(File filesDir, String subDirName, String fileName) throws IOException {
        Path subDir = Paths.get(filesDir.getAbsolutePath(), subDirName);
        if (!Files.exists(subDir)) {
            Files.createDirectories(subDir);
        }
        Path filePath = subDir.resolve(fileName);
        return filePath.toAbsolutePath().toString();
    }


    public static void restoreWeightsFromCheckpoint(Interpreter anotherInterpreter, String checkpointFilePath) {
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", checkpointFilePath);
        Map<String, Object> outputs = new HashMap<>();
        anotherInterpreter.runSignature(inputs, outputs, "restore");
    }


    public static HashMap<Integer, Integer> getOriginalLabelsToDatasetLabelsMap(HashMap<String, Integer> labelsMap) {
        Set<Integer> uniqueClasses = new HashSet<>(labelsMap.values());
        List<Integer> sortedDatasetClasses = new ArrayList<>(uniqueClasses);
        Collections.sort(sortedDatasetClasses);

        HashMap<Integer, Integer> originalLabelsToDatasetLabelsMap = new HashMap<>();
        for (int i = 0; i < sortedDatasetClasses.size(); ++i) {
            originalLabelsToDatasetLabelsMap.put(sortedDatasetClasses.get(i), i);
        }
        return originalLabelsToDatasetLabelsMap;
    }

    public static int getNumClasses(HashMap<String, Integer> labelsMap) {
        Set<Integer> uniqueClasses = new HashSet<>(labelsMap.values());
        return uniqueClasses.size();
    }

    public static int[] getModelInputShape(AssetManager assets, String modelFilename) {
        try {
            Interpreter interpreter = new Interpreter(loadModelFile(assets, modelFilename));
            int[] inputShape = interpreter.getInputTensor(0).shape();
            interpreter.close();
            return inputShape;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static List<String> getImageFileNamesFromDataset(String datasetDirAbsolutePath) {
        File dir = new File(datasetDirAbsolutePath);
        List<String> fileNames = new ArrayList<>();

        if (dir.isDirectory()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File file : files) {
                    String fileName = file.getName();
                    if (!fileName.contains(".json")) fileNames.add(fileName);
                }
            }
        }
        return fileNames;
    }

    public static HashMap<String, Integer> getLabelMapFromDataset(String datasetDirAbsolutePath) {
        try {
            Gson gson = new Gson();
            String labelsJsonAbsolutePath = new File(datasetDirAbsolutePath, "labels.json").getAbsolutePath();
            FileReader reader = new FileReader(labelsJsonAbsolutePath);

            Type type = new TypeToken<HashMap<String, Integer>>() {
            }.getType();
            HashMap<String, Integer> labelMap = gson.fromJson(reader, type);

            reader.close();

            return labelMap;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static FloatBuffer readImageAsFloatBuffer(String datasetDirAbsolutePath, String imageFileName, int imgWidth, int imgHeight) {
        try {
            String imageFileAbsolutePath = new File(datasetDirAbsolutePath, imageFileName).getAbsolutePath();
            Bitmap bitmap = BitmapFactory.decodeFile(imageFileAbsolutePath);
            if (bitmap == null) {
                throw new IOException("Failed to decode image: " + imageFileAbsolutePath);
            }

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
//                floatBuffer.put(green);
//                floatBuffer.put(blue);
            }
            floatBuffer.rewind();
            return floatBuffer;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static FloatBuffer readLabelAsFloatBuffer(int labelIndex, int numClasses) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * numClasses).order(ByteOrder.nativeOrder());
        FloatBuffer labelBuffer = byteBuffer.asFloatBuffer();
        labelBuffer.put(labelIndex, 1.0f);  // One-hot encoding
        labelBuffer.rewind();
        return labelBuffer;
    }

    public static void saveModelWeights(Interpreter interpreter, File filesDir, String newModelFileName) throws IOException {
        String ckptAbsolutePath = getAbsolutePathFromFilesDir(filesDir, "models", newModelFileName + ".ckpt");
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", ckptAbsolutePath);
        Map<String, Object> outputs = new HashMap<>();

        interpreter.runSignature(inputs, outputs, "save");
        Log.d("FinetuneActivity", "Model weights saved to " + ckptAbsolutePath);
    }
}
