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

fun downloadModelFile(downloadUrl: String, filesDir: File, fileName: String) {
    try {
        val inputStream = URL(downloadUrl).openStream()
        val outputFile = File(filesDir, "/models/$fileName.tflite")
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

suspend fun getMarketplaceFiles(): List<Model> {
    val client = OkHttpClient()
    val gson = Gson()
    val request = Request.Builder()
        .url("http://10.0.2.2:8000/weights")//10.0.2.2 refers to localhost
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