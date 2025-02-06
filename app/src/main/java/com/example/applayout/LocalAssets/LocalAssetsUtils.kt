package com.example.applayout.LocalAssets

import android.content.Context
import android.net.ConnectivityManager
import android.util.Log
import com.example.applayout.Data.Database.AppDatabase
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Data.Model.Finetune
import com.example.applayout.Data.Model.LocalModel
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.Data.Model.Model
import com.example.applayout.Models.evaluationApi
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

fun saveModelintoDB(context: Context, fileName: String): LocalModel {
    val db = AppDatabase.getDatabase(context)
    val modelDao = db.localModelDao()
    val newModel = LocalModel(
        uniqueIdentifier = fileName
    )
    modelDao.insertModel(newModel)
    logDatabaseContents(db)
    return newModel
}

fun saveDatasetintoDB(context: Context, fileName: String) {
    val db = AppDatabase.getDatabase(context)
    val datasetDao = db.localDatasetDao()
    val newDataset = Dataset(
        uniqueIdentifier = fileName

    )
    datasetDao.insertDataset(newDataset)
    logDatabaseContents(db)
}


fun createDatasetFolder(fileDir: File): String {
    val datasetsDir = File(fileDir, "datasets")
    if (!datasetsDir.exists()) {
        datasetsDir.mkdirs() // Create the datasets directory if it doesn't exist
    }
    val randomID = UUID.randomUUID()
    val newDatasetDir = File(datasetsDir, randomID.toString())
    newDatasetDir.mkdirs()
    val labelsFile = File(newDatasetDir, "labels.json")
    labelsFile.writeText("{}")
    return randomID.toString()
}


suspend fun downloadFile(fileDir: File, parentFolder: String): LocalModel? {
    val TAG = "DownloadFile"
    return withContext(Dispatchers.IO) {
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
            val newModel = LocalModel(
                uniqueIdentifier = randomID.toString()
            )
            val outputFile = File(modelsDir, randomFileName)

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "Download failed: HTTP ${response.code}")
                    return@withContext null // Exit early in case of failure
                }
                Log.d(TAG, "Download successful, saving to: ${outputFile.absolutePath}")

                response.body?.byteStream()?.use { input ->
                    FileOutputStream(outputFile).use { output ->
                        input.copyTo(output)
                    }
                }

                Log.d(TAG, "File successfully saved: ${outputFile.absolutePath}")
                return@withContext newModel // Ensure the function returns `newModel`
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file: ${e.message}", e)
            return@withContext null // Return null if an exception occurs
        }
    }
}


fun deleteLocalDirectory(directoryName: String, fileDir: File): Boolean {
    val tag = "DeleteDirectory"
    val datasetsDir = File(fileDir, "datasets")
    val specificDirectory = File(datasetsDir, directoryName)
    return if (specificDirectory.exists() && specificDirectory.isDirectory) {
        specificDirectory.deleteRecursively()
    } else {
        Log.d(tag, "Directory $directoryName does not exist.")
        false
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

fun countFilesInDirectory(fileDir: File, parentFolder: String): Int {
    return try {
        val parentDir = File(fileDir, parentFolder)
        if (parentDir.exists() && parentDir.isDirectory) {
            parentDir.listFiles()
                ?.count { it.isFile } ?: 0
        } else {
            0
        }
    } catch (e: Exception) {
        e.printStackTrace()
        0
    }
}


fun listLocalResources(fileDir: File, parentFolder: String, isFile: Boolean): List<String> {
    return try {
        val parentDir = File(fileDir, parentFolder)
        if (parentDir.exists() && parentDir.isDirectory) {
            parentDir.listFiles()
                ?.filter { if (isFile) it.isFile else it.isDirectory }
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
    val notUploadedModels: List<Model>
)

suspend fun fetchModelOwner(modelUniqueIdentifier: String, username: String): Boolean {
    val client = OkHttpClient()
    val gson = Gson()
    return withContext(Dispatchers.IO) {

        try {
//            val url = "http://10.0.2.2:8000/weights/$modelUniqueIdentifier/user"
            val url = "https://android-p2p-backend.onrender.com/weights/$modelUniqueIdentifier/user"
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
    val fileNames = listLocalResources(filesDir, "models", true)
    val uploadedModels = mutableListOf<Model>()
    val notUploadedModels = mutableListOf<Model>()
    val client = OkHttpClient()
    val gson = Gson()
    withContext(Dispatchers.IO) {
        fileNames.map{ it.substringBeforeLast(".") } //remove file extensions
            .forEach { fileName ->
            try {
//                val url = "http://10.0.2.2:8000/weights/$fileName"
                val url = "https://android-p2p-backend.onrender.com/weights/$fileName"
                val request = Request.Builder().url(url).build()
                val response = client.newCall(request).execute()
                if (response.isSuccessful) {
                    val body = response.body?.string()
                    val model = gson.fromJson<Model>(body, object : TypeToken<Model>() {}.type)
                    if (model != null && model.is_uploaded) {
                        uploadedModels.add(model)
                    } else if (model != null) { //!model.is_uploaded
                        model.isOwner = true
                        notUploadedModels.add(model)
                    }
                }
            } catch (e: Exception) {
                Log.e("FetchModels", "Exception: ${e.message}", e)
            }
        }
    }
    return ModelResponse(uploadedModels, notUploadedModels)
}

data class RemoteDataset(
    val uniqueIdentifier: String,
    var model_task: String,
    var description: String,
    var class_labels: List<String>,
    var isUploaded: Boolean = false
) {
    fun toDataset(): Dataset {
        return Dataset(
            uniqueIdentifier = uniqueIdentifier,
            model_task = model_task,
            description = description ?: "",
            class_labels = class_labels.joinToString(", "),
            isUploaded = isUploaded
        )
    }
}

suspend fun fetchDatasetInfo(localDatasetList: List<String>): List<Dataset> {
    val datasetDataList = mutableListOf<Dataset>()
    val client = OkHttpClient()
    val gson = Gson()
    withContext(Dispatchers.IO) {
        localDatasetList
            .forEach { datasetName ->
                try {
                    val url = "https://android-p2p-backend.onrender.com/dataset/$datasetName"
                    val request = Request.Builder().url(url).build()
                    val response = client.newCall(request).execute()
                    if (response.isSuccessful) {
                        val body = response.body?.string()
                        val dataset = gson.fromJson<RemoteDataset>(
                            body,
                            object : TypeToken<RemoteDataset>() {}.type
                        )
                        if (dataset != null) {
                            datasetDataList.add(dataset.toDataset())
                        } else {
                            Log.e("fetchDataset", "invalid data fetched")
                        }
                    }
                } catch (e: Exception) {
                    Log.e("fetchDataset", "Exception: ${e.message}", e)
                }
            }
    }
    Log.d("fetchDataset", localDatasetList.toString())
    Log.d("fetchDataset", datasetDataList.toString())
    return datasetDataList
}


data class createModelPayload(
    val weight: Model,
    val username: String
)

suspend fun editModel(modelUniqueIdentifier: String, formData: Model) {
    val client = OkHttpClient()
    val gson = Gson()
    val payload = gson.toJson(formData)
    Log.d("EditModel", "Generated JSON Payload: $payload")

    val requestBody = payload.toRequestBody("application/json".toMediaType())
    val request = Request.Builder()
//        .url("http://10.0.2.2:8000/weights/${modelUniqueIdentifier}")
        .url("https://android-p2p-backend.onrender.com/weights/${modelUniqueIdentifier}")
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

suspend fun uploadModelNode(
    filesDir: File,
    localModelData: LocalModel,
    username: String
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val modelFile = File(filesDir, "/models/${localModelData.uniqueIdentifier}.tflite")

        val modelData = Model(
            uniqueIdentifier = localModelData.uniqueIdentifier,
            model_task = localModelData.model_task,
            last_trained = System.currentTimeMillis().toString(),
            description = localModelData.description,
            weight_size = modelFile.length(),
        )

        val payload = gson.toJson(createModelPayload(weight = modelData, username = username))
        Log.d("uploadModelNode", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
//            .url("http://10.0.2.2:8000/weights/create")
            .url("https://android-p2p-backend.onrender.com/weights/create")
            .post(requestBody)
            .build()

        Log.d("uploadModelNode", "Sending POST request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "uploadModelNode",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("uploadModelNode", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}

data class uploadModelPayload(
    val uniqueIdentifier: String,
    val is_uploaded: Boolean = false,
    val public_link: String = ""
)

suspend fun uploadModel(
    filesDir: File,
    modelUniqueIdentifier: String,
    context: Context
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val modelFile = File(filesDir, "/models/${modelUniqueIdentifier}.tflite")

        Log.d("UploadModel", "Preparing to upload model: $modelUniqueIdentifier")
        Log.d("UploadModel", "File exists: ${modelFile.exists()}, File size: ${modelFile.length()}")

        val publicLink = uploadFileToGCS(modelUniqueIdentifier, modelFile, context)
        Log.d("UploadModel", "File uploaded to GCS. Public Link: $publicLink")


        val payload = gson.toJson(
            uploadModelPayload(
                uniqueIdentifier = modelUniqueIdentifier,
                public_link = publicLink,
                is_uploaded = true
            )
        )
        Log.d("UploadModel", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
//        val request = Request.Builder()
////            .url("http://10.0.2.2:8000/weights/create")
//            .url("https://android-p2p-backend.onrender.com/weights/create")
//            .post(requestBody)
//            .build()

        val request = Request.Builder()
//        .url("http://10.0.2.2:8000/weights/${modelUniqueIdentifier}")
            .url("https://android-p2p-backend.onrender.com/weights/${modelUniqueIdentifier}")
            .patch(requestBody)
            .build()

        Log.d("UploadModel", "Sending PATCH request to serverUrl")
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
data class DatasetForApi(
    val uniqueIdentifier: String,
    val model_task: String,
    val description: String,
    val num_images: Int,
    val class_labels: List<String>
)


data class UploadDatasetPayload(
    val dataset: DatasetForApi,
    val username: String
)

suspend fun uploadDatasetNode(
    datasetData: Dataset,
    username: String,
    filesDir: File
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val numImages = countFilesInDirectory(filesDir, "datasets/${datasetData.uniqueIdentifier}")
        val datasetForApi = DatasetForApi(
            uniqueIdentifier = datasetData.uniqueIdentifier,
            model_task = datasetData.model_task,
            description = datasetData.description,
            num_images = numImages - 1, //remove labels.json count
            class_labels = datasetData.getClassLabelsAsList()
        )

        val payload =
            gson.toJson(UploadDatasetPayload(dataset = datasetForApi, username = username))
        Log.d("uploadDataset", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
//            .url("http://10.0.2.2:8000/weights/create")
            .url("https://android-p2p-backend.onrender.com/dataset/create")
            .post(requestBody)
            .build()

        Log.d("uploadDataset", "Sending POST request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "uploadDataset",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("UploadModel", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}


suspend fun editDataset(
    datasetData: Dataset,
    filesDir: File
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val numImages = countFilesInDirectory(filesDir, "datasets/${datasetData.uniqueIdentifier}")
        val payload =
            gson.toJson(
                DatasetForApi(
                    uniqueIdentifier = datasetData.uniqueIdentifier,
                    model_task = datasetData.model_task,
                    description = datasetData.description,
                    num_images = numImages - 1, //remove labels.json count
                    class_labels = datasetData.getClassLabelsAsList()
                )
            )

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("https://android-p2p-backend.onrender.com/dataset/${datasetData.uniqueIdentifier}")
            .patch(requestBody)
            .build()
        Log.d("editDataset", "Sending patch request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "editDataset",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("editDataset", "Error during dataset upload: ${e.message}", e)
        e.printStackTrace()
    }
}


data class uploadEvaluationPayload(
    val dataset_id: String,
    val weight_id: String,
    val evaluate: evaluationApi
)

suspend fun uploadEvaluationResults(weightId: String, datasetId: String, evaluate: evaluationApi) {
    try {
        val client = OkHttpClient()
        val gson = Gson()

        val payload =
            gson.toJson(
                uploadEvaluationPayload(
                    evaluate = evaluate,
                    dataset_id = datasetId,
                    weight_id = weightId
                )
            )
        Log.d("uploadResults", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
//            .url("http://10.0.2.2:8000/weights/create")
            .url("https://android-p2p-backend.onrender.com/weights/evaluation")
            .post(requestBody)
            .build()

        Log.d("uploadResults", "Sending POST request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "uploadResults",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("uploadResults", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}

data class UploadFederatedLearningRelationshipPayload(
    val resultant_id: String,
    val component_ids: List<String>
)

suspend fun uploadFederatedLearningRelationship(
    relationshipData: LocalRelationship,
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
            val payload = gson.toJson(
                UploadFederatedLearningRelationshipPayload(
                    resultant_id = relationshipData.modelUniqueIdentifier,
                    component_ids = relationshipData.sourceUniqueIdentifiers.split(",")
                )
            )
            Log.d("uploadRelationship", "Generated JSON Payload: $payload")
            val requestBody = payload.toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
//                .url("http://10.0.2.2:8000/weights/combine")
                .url("https://android-p2p-backend.onrender.com/weights/combine")
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
    } catch (e: Exception) {
        Log.e("UploadModel", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}

data class UploadFinetuningRelationshipPayload(
    val dataset_id: String,
    val weight_id: String,
    val finetune: Finetune
)


suspend fun uploadFinetuningRelationship(
    relationshipData: LocalRelationship,
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val finetune = Finetune()
        val payload = gson.toJson(
            UploadFinetuningRelationshipPayload(
                weight_id = relationshipData.modelUniqueIdentifier,
                dataset_id = relationshipData.sourceUniqueIdentifiers,
                finetune = finetune
            )
        )
        Log.d("uploadRelationship", "Generated JSON Payload: $payload")
        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("https://android-p2p-backend.onrender.com/weights/finetuned")
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
        Log.d(tag, "Model Task: ${model.model_task}")

    }
    database.localDatasetDao().getAllDatasets().forEach { dataset ->
        Log.d(tag, "dataset ID: ${dataset.uniqueIdentifier}")
        Log.d(tag, "dataset Description: ${dataset.description}")
        Log.d(tag, "dataset Task: ${dataset.model_task}")

    }

}

suspend fun checkModelExistence(ids: List<String>): List<String> {
    val missingIds = mutableListOf<String>()
    val client = OkHttpClient()
    for (id in ids) {
        try {
            val request = Request.Builder()
//                .url("http://10.0.2.2:8000/weights/$id")
                .url("https://android-p2p-backend.onrender.com/weights/$id")
                .get()
                .build()
            val response = withContext(Dispatchers.IO) {
                client.newCall(request).execute()
            }
            if (!response.isSuccessful) {
                missingIds.add(id)
            }
        } catch (e: Exception) {
            Log.e("ModelCheck", "Error checking ID $id: ${e.message}")
            missingIds.add(id)
        }
    }
    return missingIds
}

fun isInternetConnected(context: Context): Boolean {
    val connectivityManager =
        context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    val activeNetworkInfo = connectivityManager.activeNetworkInfo
    return activeNetworkInfo != null && activeNetworkInfo.isConnected
}