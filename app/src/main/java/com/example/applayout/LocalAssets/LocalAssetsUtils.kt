package com.example.applayout.LocalAssets

import com.example.applayout.Data.Model.Model
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import okhttp3.OkHttpClient
import okhttp3.Request
import com.google.gson.reflect.TypeToken
import android.util.Log

fun getLocalFiles(fileDir: File, parentFolder: String): List<String> {
    return try {
        val modelsDir = File(fileDir, parentFolder)
        if (modelsDir.exists() && modelsDir.isDirectory) {
            modelsDir.listFiles()
                ?.filter { it.isFile }
                ?.map { it.name }
                ?: emptyList()
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        emptyList()
    }
}
data class ModelResponse(
    val uploadedModels: List<Model>,
    val notUploadedModels: List<String>
)

suspend fun fetchModelsInfo(fileNames: List<String>): ModelResponse  {
    val uploadedModels = mutableListOf<Model>()
    val notUploadedModels = mutableListOf<String>()
    val client = OkHttpClient()
    val gson = Gson()
    withContext(Dispatchers.IO) {
        fileNames.map{ it.substringBeforeLast(".") } //remove file extensions
            .forEach { fileName ->
            try {
                val url = "http://192.168.1.5:8000/weights/$fileName"
                val request = Request.Builder().url(url).build()
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    val body = response.body?.string()
                    val model = gson.fromJson<Model>(body, object : TypeToken<Model>() {}.type)
                    if (model != null) {
                        uploadedModels.add(model)
                    } else {
                        notUploadedModels.add(fileName)
                    }
                } else {
                    Log.e("FetchModels", "Error: ${response.code}")
                    notUploadedModels.add(fileName)
                }
            } catch (e: Exception) {
                Log.e("FetchModels", "Exception: ${e.message}", e)
            }
        }
    }
    return ModelResponse(uploadedModels, notUploadedModels)
}