package com.example.applayout.Models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.compose.ui.graphics.asImageBitmap
import com.example.applayout.Finetune.FinetuneUtils
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln

data class evaluationApi(
    val accuracy: Double,
    val precision: Double,
    val class_performance: List<Double>,
    val evaluationDate: String
)

fun readFloatsFromFile(context: Context, fileName: String): List<Float> {
    val file = File(context.filesDir, fileName)
    return try {
        val content = file.readText().trim()
        Gson().fromJson(content, object : TypeToken<List<Float>>() {}.type) ?: emptyList()
    } catch (e: Exception) {
        e.printStackTrace()
        emptyList()
    }
}


fun saveFile(context: Context, fileName: String, data: Any) {
    val tag = "ModelUtils"
    try {
        val file = File(context.filesDir, fileName)
        FileOutputStream(file).use { fos ->
            val content = when (data) {
                is String -> data
                else -> Gson().toJson(data) // Convert objects/lists to JSON
            }
            fos.write(content.toByteArray(Charsets.UTF_8))
        }
        Log.d(tag, "File saved: ${file.absolutePath}")
    } catch (e: Exception) {
        e.printStackTrace()
        Log.d(tag, "Failed to save file: ${e.message}")
    }
}

fun softmax(logits: FloatArray): FloatArray {
    val expValues = logits.map { exp(it) }
    val sumExp = expValues.sum()
    return expValues.map { it / sumExp }.toFloatArray()
}

//[[predict, correct],[predict, correct]]
fun runInferenceOnDirectory(
    context: Context,
    modelId: String,
    datasetId: String
): List<Pair<Int, Int>> {
    val relativeDirectoryPath = "datasets/${datasetId}"
    val interpreter = Interpreter(
        FinetuneUtils.loadModelFile(
            context.assets,
            "model.tflite"
        )
    )
    val modelFileAbsolutePath =
        FinetuneUtils.getAbsolutePathFromFilesDir(context.filesDir, "models", "$modelId.ckpt")
    val modelFile = File(modelFileAbsolutePath)
    if (!modelFile.exists()) {
        Log.d("Inference", "Model file does not exist: $modelFileAbsolutePath")
        return emptyList()
    }
    FinetuneUtils.restoreWeightsFromCheckpoint(interpreter, modelFileAbsolutePath)


    val directory = File(context.filesDir, relativeDirectoryPath)
    if (!directory.exists() || !directory.isDirectory) {
        Log.d("Inference", "Directory does not exist or is not a directory: $relativeDirectoryPath")
        return emptyList()
    }
    val datasetDir = File(context.filesDir, "datasets/${datasetId}")
    val datasetLabels = readLabels(datasetDir)
    val results = mutableListOf<Pair<Int, Int>>()
    val inputSize = 28 // Model expects 28x28 single-channel input
    val outputTensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10), DataType.FLOAT32)
    val losses = mutableListOf<Float>()

    directory.listFiles()?.filter { it.isFile && it.extension in listOf("jpg", "png", "jpeg") }
        ?.forEach { imageFile ->
            val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
            if (bitmap != null) {
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
                val grayscaleBitmap = convertToGrayscale(resizedBitmap)
                val inputTensor = preprocessImageForSingleChannel(grayscaleBitmap, inputSize)
                interpreter.run(inputTensor, outputTensorBuffer.buffer.rewind())
                val outputArray = outputTensorBuffer.floatArray
                val predictedLabel = outputArray.indices.maxByOrNull { outputArray[it] } ?: -1
                val actualLabel = datasetLabels[imageFile.name]?.toIntOrNull() ?: -1

                val probabilities = softmax(outputArray)

                // Compute loss (Cross-Entropy)
                val actualProbability =
                    if (actualLabel in probabilities.indices) probabilities[actualLabel] else 0f
                Log.d("ModelUtilsEval", "actualProbability: $actualProbability")
                val loss =
                    if (actualProbability > 0) -ln(actualProbability) else Float.POSITIVE_INFINITY
                Log.d("ModelUtilsEval", "loss: $loss")
                losses.add(loss)

                results.add(Pair(predictedLabel, actualLabel))
            } else {
                println("Failed to decode image: ${imageFile.name}")
            }
        }

    saveFile(context, "last_run_eval_losses", losses)

    interpreter.close()
    return results
}

private fun preprocessImageForSingleChannel(bitmap: Bitmap, inputSize: Int): ByteBuffer {
    val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize) // Float32 (4 bytes)
    inputBuffer.order(ByteOrder.nativeOrder())

    val intValues = IntArray(inputSize * inputSize)
    bitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

    // Normalize image data
    for (pixelValue in intValues) {
        val grayscaleValue = ((pixelValue shr 16) and 0xFF) / 255.0f // Only need one channel (R)
        inputBuffer.putFloat(grayscaleValue)
    }
    return inputBuffer
}

private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
    val width = bitmap.width
    val height = bitmap.height
    val grayscaleBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    for (y in 0 until height) {
        for (x in 0 until width) {
            val pixel = bitmap.getPixel(x, y)
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8 and 0xFF)
            val b = (pixel and 0xFF)
            val gray = (r * 0.3 + g * 0.59 + b * 0.11).toInt()
            val grayPixel = (0xFF shl 24) or (gray shl 16) or (gray shl 8) or gray
            grayscaleBitmap.setPixel(x, y, grayPixel)
        }
    }
    return grayscaleBitmap
}


fun readLabels(datasetDir: File): Map<String, String> {
    val labelsFile = File(datasetDir, "labels.json")
    return if (labelsFile.exists()) {
        try {
            val jsonContent = labelsFile.readText()
            val type = object : TypeToken<Map<String, String>>() {}.type
            Gson().fromJson(jsonContent, type)
        } catch (e: Exception) {
            e.printStackTrace()
            emptyMap()
        }
    } else {
        emptyMap()
    }
}

fun readLabelsDefault(datasetDir: File): Map<String, String> {
    val labelsFile = File(datasetDir, "labels.json")
    val existingLabels: MutableMap<String, String> = try {
        if (labelsFile.exists()) {
            val jsonContent = labelsFile.readText()
            val type = object : TypeToken<Map<String, String>>() {}.type
            Gson().fromJson<Map<String, String>>(jsonContent, type)?.toMutableMap()
                ?: mutableMapOf()
        } else {
            mutableMapOf()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        mutableMapOf()
    }

    // Get all image filenames in the dataset directory
    val allFiles = datasetDir.listFiles()?.map { it.name }?.filter { it != "labels.json" }?.toSet()
        ?: emptySet()

    // Ensure all images have a label, defaulting to "0"
    var updated = false
    allFiles.forEach { fileName ->
        if (!existingLabels.containsKey(fileName)) {
            existingLabels[fileName] = "0"  // Assign default label
            updated = true
        }
    }

    // If updates were made, write back to labels.json
    if (updated) {
        labelsFile.writeText(Gson().toJson(existingLabels))
    }

    return existingLabels
}

fun loadImageBitmap(filePath: String): androidx.compose.ui.graphics.ImageBitmap {
    val bitmap = BitmapFactory.decodeFile(filePath)
    return bitmap?.asImageBitmap() ?: androidx.compose.ui.graphics.ImageBitmap(1, 1)
}