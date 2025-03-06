package com.example.applayout.Finetune;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

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
import java.util.HashMap;
import java.util.Map;

public class FinetuneUtils {

    public static MappedByteBuffer loadModelFile(String modelFileAbsolutePath)
            throws IOException {
//        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
//        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
//        FileChannel fileChannel = inputStream.getChannel();
//        long startOffset = fileDescriptor.getStartOffset();
//        long declaredLength = fileDescriptor.getDeclaredLength();
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        File file = new File(modelFileAbsolutePath);
        FileInputStream inputStream = new FileInputStream(file);
        FileChannel fileChannel = inputStream.getChannel();
        long fileSize = fileChannel.size();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
    }

    public static FloatBuffer readImageAsFloatBuffer(Context context, String assetPath, int imgWidth, int imgHeight) {
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

    public static FloatBuffer readLabelAsFloatBuffer(Context context, String assetPath, int numClasses) {
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

    public static void saveModelWeights(Context context, Interpreter interpreter) {
        String filename = "trained_model_weights.ckpt";
        File file = new File(context.getFilesDir(), filename);
        Map<String, Object> inputs = new HashMap<>();
        inputs.put("checkpoint_path", file.getAbsolutePath());
        Map<String, Object> outputs = new HashMap<>();

        interpreter.runSignature(inputs, outputs, "save");
        Log.d("FinetuneActivity", "Model weights saved to " + file.getAbsolutePath());
    }
}
