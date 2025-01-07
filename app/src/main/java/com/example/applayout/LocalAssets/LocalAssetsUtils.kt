package com.example.applayout.LocalAssets

import android.content.Context
import android.util.Log
import com.example.applayout.Data.Database.AppDatabase
import com.example.applayout.Data.Model.LocalModel
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.Data.Model.Model
import com.example.applayout.R
import com.google.auth.oauth2.ServiceAccountCredentials
import com.google.cloud.storage.BlobId
import com.google.cloud.storage.BlobInfo
import com.google.cloud.storage.Storage
import com.google.cloud.storage.StorageOptions
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.util.UUID


fun saveFile(context: Context, fileName: String) {
    val db = AppDatabase.getDatabase(context)
    val modelDao = db.localModelDao()
    val newModel = LocalModel(
        uniqueIdentifier = fileName
    )
    modelDao.insertModel(newModel)
    logDatabaseContents(db)
}

suspend fun downloadFile(context: Context, fileDir: File, parentFolder: String) {
    val TAG = "DownloadFile"
    withContext(Dispatchers.IO) {
        try {
            val client = OkHttpClient()
            val fileUrl = "https://storage.googleapis.com/android-p2p/weights/test.tflite"

            Log.d(TAG, "Starting download from URL: $fileUrl")

            val request = Request.Builder().url(fileUrl).build()

            val modelsDir = File(fileDir, parentFolder)
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }
            val randomID = UUID.randomUUID()
            val randomFileName = "${randomID}.tflite"
            saveFile(context, randomID.toString()) //save an entry into db
            val outputFile = File(modelsDir, randomFileName)
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "Download failed: HTTP ${response.code}")
                    return@use
                }
                Log.d(TAG, "Download successful, saving to: ${outputFile.absolutePath}")

                val inputStream = response.body?.byteStream()
                val outputStream = FileOutputStream(outputFile)

                inputStream?.use { input ->
                    outputStream.use { output ->
                        input.copyTo(output)
                    }
                }

                Log.d(TAG, "File successfully saved: ${outputFile.absolutePath}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file: ${e.message}", e)
        }
    }
}

fun deleteLocalFile(fileName: String, fileDir: File, parentFolder: String): Boolean {
    val tag = "DeleteFile"
    val modelsDir = File(fileDir, parentFolder)
    val modelFilePath = File(modelsDir, "${fileName}.tflite")
    return if (modelFilePath.exists()) {
        modelFilePath.delete()
    } else {
        Log.d(tag, "File $fileName does not exist.")
        false
    }
}

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

suspend fun fetchModelOwner(modelUniqueIdentifier: String, username: String): Boolean {
    val client = OkHttpClient()
    val gson = Gson()
    return withContext(Dispatchers.IO) {

        try {
            val url = "http://10.0.2.2:8000/weights/$modelUniqueIdentifier/user"
            val request = Request.Builder().url(url).build()
            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val body = response.body?.string()
                val jsonObject = gson.fromJson(body, Map::class.java)
                val modelOwner = jsonObject["username"] as? String
                return@withContext modelOwner == username
            }
            return@withContext false
        } catch (e: Exception) {
            Log.e("FetchOwner", "Exception: ${e.message}", e)
            return@withContext false
        }
    }
}

suspend fun fetchModelsInfo(filesDir: File): ModelResponse {
    val fileNames = getLocalFiles(filesDir, "models")
    val uploadedModels = mutableListOf<Model>()
    val notUploadedModels = mutableListOf<String>()
    val client = OkHttpClient()
    val gson = Gson()
    withContext(Dispatchers.IO) {
        fileNames.map{ it.substringBeforeLast(".") } //remove file extensions
            .forEach { fileName ->
            try {
                val url = "http://10.0.2.2:8000/weights/$fileName"
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
//                    Log.e("FetchModels", "Error: ${response.code}")
                    notUploadedModels.add(fileName)
                }
            } catch (e: Exception) {
                Log.e("FetchModels", "Exception: ${e.message}", e)
            }
        }
    }
    return ModelResponse(uploadedModels, notUploadedModels)
}


data class UploadModelPayload(
    val weight: Model,
    val username: String
)

suspend fun editModel(modelUniqueIdentifier: String, formData: LocalModel) {
    val client = OkHttpClient()
    val gson = Gson()
    val payload = gson.toJson(formData)
    Log.d("EditModel", "Generated JSON Payload: $payload")

    val requestBody = payload.toRequestBody("application/json".toMediaType())
    val request = Request.Builder()
        .url("http://10.0.2.2:8000/weights/${modelUniqueIdentifier}")
        .patch(requestBody)
        .build()

    Log.d("EditModel", "Sending PATCH request to serverUrl")
    withContext(Dispatchers.IO) {
        val response = client.newCall(request).execute()
        Log.d(
            "EditModel",
            "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
        )
    }
}

suspend fun uploadModel(
    filesDir: File,
    localModelData: LocalModel,
    fileName: String,
    username: String,
    context: Context
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val modelFile = File(filesDir, "/models/$fileName.tflite")

        Log.d("UploadModel", "Preparing to upload model: $fileName")
        Log.d("UploadModel", "File exists: ${modelFile.exists()}, File size: ${modelFile.length()}")

        val publicLink = uploadFileToGCS(fileName, modelFile, context)
        Log.d("UploadModel", "File uploaded to GCS. Public Link: $publicLink")

        val modelData = Model(
            uniqueIdentifier = fileName,
            model_task = localModelData.model_task,
            last_trained = System.currentTimeMillis().toString(), // Current timestamp
            description = localModelData.description,
            weight_size = modelFile.length(),
            public_link = publicLink
        )

        val payload = gson.toJson(UploadModelPayload(weight = modelData, username = username))
        Log.d("UploadModel", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("http://10.0.2.2:8000/weights/create")
            .post(requestBody)
            .build()

        Log.d("UploadModel", "Sending POST request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "UploadModel",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("UploadModel", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}


data class UploadRelationshipPayload(
    val resultant_id: String,
    val component_ids: List<String>
)

suspend fun uploadRelationship(
    relationshipData: LocalRelationship,
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        if (relationshipData.relationshipType == "Model") {
            val payload = gson.toJson(
                UploadRelationshipPayload(
                    resultant_id = relationshipData.modelUniqueIdentifier,
                    component_ids = relationshipData.sourceUniqueIdentifiers.split(",")
                )
            )
            Log.d("uploadRelationship", "Generated JSON Payload: $payload")
            val requestBody = payload.toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url("http://10.0.2.2:8000/weights/combine")
                .post(requestBody)
                .build()

            Log.d("uploadRelationship", "Sending POST request to serverUrl")
            withContext(Dispatchers.IO) {
                val response = client.newCall(request).execute()
                Log.d(
                    "uploadRelationship",
                    "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
                )
            }
        } else {
            Log.d("uploadRelationship", "not of type model")
        }
    } catch (e: Exception) {
        Log.e("UploadModel", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}

suspend fun uploadFileToGCS(
    fileName: String,
    localFile: File,
    context: Context
): String {
    return withContext(Dispatchers.IO) {
        try {
            Log.d("UploadFileToGCS", "Starting upload to Google Cloud Storage...")
            val bucketName = "android-p2p"
            val destinationPath = "weights/$fileName.tflite"
            Log.d("UploadFileToGCS", "Bucket: $bucketName, Destination Path: $destinationPath")

            val inputStream =
                context.resources.openRawResource(R.raw.android_p2p_444502_a45dbd1e7f35)
            val credentials = ServiceAccountCredentials.fromStream(inputStream)
            Log.d("UploadFileToGCS", "Loaded service account credentials successfully")

            val storage: Storage = StorageOptions.newBuilder()
                .setCredentials(credentials)
                .build()
                .service

            val blobId = BlobId.of(bucketName, destinationPath)
            val blobInfo = BlobInfo.newBuilder(blobId)
                .setContentType("application/octet-stream")
                .build()

            Log.d("UploadFileToGCS", "Creating blob: $blobId")

            FileInputStream(localFile).use { fileInputStream ->
                val writer = storage.writer(blobInfo)
                val buffer = ByteArray(1024)
                val byteBuffer = ByteBuffer.allocate(1024)
                var bytesRead: Int

                Log.d("UploadFileToGCS", "Uploading file: ${localFile.absolutePath}")
                while (fileInputStream.read(buffer).also { bytesRead = it } != -1) {
                    byteBuffer.clear() // Reset the ByteBuffer
                    byteBuffer.put(buffer, 0, bytesRead) // Wrap byte[] into ByteBuffer
                    byteBuffer.flip() // Prepare ByteBuffer for writing
                    writer.write(byteBuffer)
                }
                writer.close()
                Log.d("UploadFileToGCS", "File upload completed")
            }

            val publicUrl = "https://storage.googleapis.com/$bucketName/$destinationPath"
            Log.d("UploadFileToGCS", "Public URL: $publicUrl")

            publicUrl
        } catch (e: Exception) {
            Log.e("UploadFileToGCS", "Error uploading file to GCS: ${e.message}", e)
            e.printStackTrace()
            ""
        }
    }
}

fun logDatabaseContents(database: AppDatabase) {
    val tag = "DatabaseContents"
    database.localModelDao().getAllModels().forEach { model ->
        Log.d(tag, "Model ID: ${model.uniqueIdentifier}")
        Log.d(tag, "Model Description: ${model.description}")
        Log.d(tag, "FILE Path: ${model.model_task}")

    }

}
