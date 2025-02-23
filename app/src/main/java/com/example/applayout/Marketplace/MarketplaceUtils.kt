package com.example.applayout.Marketplace

import android.util.Log
import com.example.applayout.Data.Model.Model
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.URL

val TAG = "MarketplaceUtils"
private const val BASE_URL = "10.96.181.80"
//          10.0.2.2 refers to localhost
//        .url("https://android-p2p-backend.onrender.com/weights")

suspend fun downloadModelFile(downloadUrl: String, filesDir: File, fileName: String) {
    withContext(Dispatchers.IO) {
        try {
            // Ensure the /models directory exists
            val modelsDir = File(filesDir, "models")
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }

            val outputFile = File(modelsDir, "$fileName.tflite")
            if (outputFile.exists()) {
                Log.d(TAG, "File already exists: ${outputFile.absolutePath}")
                return@withContext
            }

            // Download the file
            val inputStream = URL(downloadUrl).openStream()
            val outputStream = FileOutputStream(outputFile)
            inputStream.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "File downloaded successfully: ${outputFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file: ${e.message}", e)
            e.printStackTrace()
        }
    }
}

suspend fun getMarketplaceFiles(): List<Model> {
    val client = OkHttpClient()
    val gson = Gson()
    val request = Request.Builder()
        .url("http://$BASE_URL:8000/weights")
        .get()
        .build()
    return try {
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            val json = response.body?.string()
            if (response.isSuccessful) {
                val listType = object : TypeToken<List<Model>>() {}.type
                val models = gson.fromJson<List<Model>>(json, listType)
                models ?: emptyList()
            } else {
                Log.e(TAG, "Error: HTTP ${response.code} - ${response.message}")
                emptyList()
            }
        }
    } catch (e: IOException) {
        Log.e(TAG, "Exception occurred during network call: ${e.message}", e)
        emptyList()
    }
}

suspend fun increment(modelUniqueIdentifier: String, isUsage: Boolean) {
    val client = OkHttpClient()
    withContext(Dispatchers.IO) {
        try {
            val url = if (isUsage)
                "http://$BASE_URL:3000/weights/$modelUniqueIdentifier/increment-usage"
            else
                "http://$BASE_URL:3000/weights/$modelUniqueIdentifier/increment-likes"
            val request = Request.Builder()
                .url(url)
                .patch(okhttp3.RequestBody.create(null, ByteArray(0))) // Empty PATCH body
                .build()
            client.newCall(request).execute()

        } catch (e: Exception) {
            Log.e("Increment", "Exception: ${e.message}", e)
        }
    }
}