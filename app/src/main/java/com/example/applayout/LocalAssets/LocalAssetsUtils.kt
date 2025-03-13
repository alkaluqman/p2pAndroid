package com.example.applayout.LocalAssets

import android.content.Context
import android.net.ConnectivityManager
import android.os.Build
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
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.URL
import java.nio.ByteBuffer
import java.util.UUID
import java.util.zip.ZipInputStream

//private const val BACKEND_URL = "10.96.181.80"
//private const val BACKEND_URL = "192.168.2.159"
private const val BACKEND_URL = "android-p2p-backend.onrender.com"

fun getNewModel(id: String? = null): LocalModel {
    val randomID = UUID.randomUUID()
    val newModel = LocalModel(
        uniqueIdentifier = id ?: randomID.toString(),
    )
    return newModel
}

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


suspend fun createDatasetFolder(fileDir: File, username: String) {
    val datasetsDir = File(fileDir, "datasets")
    if (!datasetsDir.exists()) {
        datasetsDir.mkdirs() // Create the datasets directory if it doesn't exist
    }
    val randomID = UUID.randomUUID()
    val newDatasetDir = File(datasetsDir, randomID.toString())
    newDatasetDir.mkdirs()
    val labelsFile = File(newDatasetDir, "labels.json")
    labelsFile.writeText("{}")
    uploadDatasetNode(Dataset(randomID.toString()), username, fileDir)
}

suspend fun populateDataset(fileDir: File, username: String) {
    val datasetsDir = File(fileDir, "datasets")
    if (!datasetsDir.exists()) {
        datasetsDir.mkdirs() // Create the datasets directory if it doesn't exist
    }
    val randomID = UUID.randomUUID()
    val newDatasetDir = File(datasetsDir, randomID.toString())
    newDatasetDir.mkdirs()
    val zipFile = File(newDatasetDir, "dataset.zip")
    downloadFile("https://storage.googleapis.com/android-p2p/weights/dataset.zip", zipFile)
    unzip(zipFile, newDatasetDir)

    zipFile.delete()
    uploadDatasetNode(Dataset(randomID.toString()), username, fileDir)
}

suspend fun downloadFile(url: String, outputFile: File) = withContext(Dispatchers.IO) {
    URL(url).openStream().use { input ->
        FileOutputStream(outputFile).use { output ->
            input.copyTo(output)
        }
    }
}

fun unzip(zipFile: File, targetDir: File) {
    ZipInputStream(BufferedInputStream(FileInputStream(zipFile))).use { zis ->
        var entry = zis.nextEntry
        while (entry != null) {
            val outFile = File(targetDir, entry.name)
            if (entry.isDirectory) {
//                outFile.mkdirs()
                Log.d("Unzip", "Skipping directory entry: ${entry.name}")
            } else {
                outFile.parentFile?.mkdirs()
                FileOutputStream(outFile).use { output ->
                    zis.copyTo(output)
                }
            }
            zis.closeEntry()
            entry = zis.nextEntry
        }
    }
}


suspend fun downloadFile(fileDir: File, parentFolder: String): LocalModel? {
    val TAG = "DownloadFile"
    return withContext(Dispatchers.IO) {
        try {
            val client = OkHttpClient()
            val fileUrl = "https://storage.googleapis.com/android-p2p/weights/test.ckpt"

            Log.d(TAG, "Starting download from URL: $fileUrl")

            val request = Request.Builder().url(fileUrl).build()

            val modelsDir = File(fileDir, parentFolder)
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }

            val randomID = UUID.randomUUID()
            val randomFileName = "${randomID}.ckpt"
            val outputFile = File(modelsDir, randomFileName)
            val newModel = LocalModel(
                uniqueIdentifier = randomID.toString(),
                absoluteFilePath = outputFile.absolutePath
            )

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
    val modelFilePath = File(modelsDir, "${fileName}.ckpt")
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
//            val url = "http://$BASE_URL:8000/weights/$modelUniqueIdentifier/user"
            val url = "https://$BACKEND_URL/weights/$modelUniqueIdentifier/user"
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
    val modelDir = File(filesDir, "models")
    val modelFileExt = ".ckpt"
    withContext(Dispatchers.IO) {
        fileNames.map { it.substringBeforeLast(".") } //remove file extensions
            .forEach { fileName ->
                try {
//                    val url = "http://$BASE_URL:8000/weights/$fileName"
                    val url = "https://$BACKEND_URL/weights/$fileName"
                    val request = Request.Builder().url(url).build()
                    val response = client.newCall(request).execute()
                    if (response.isSuccessful) {
                        val body = response.body?.string()
                        val model = gson.fromJson<Model>(body, object : TypeToken<Model>() {}.type)
                        if (model != null) {
                            model.absoluteFilePath =
                                modelDir.resolve(fileName + modelFileExt).absolutePath
                            if (model.is_uploaded) {
                                uploadedModels.add(model)
                            } else { //!model.is_uploaded
                                model.isOwner = true
                                notUploadedModels.add(model)
                            }
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

suspend fun fetchDatasetInfo(filesDir: File, localDatasetList: List<String>): List<Dataset> {
    val datasetDataList = mutableListOf<Dataset>()
    val client = OkHttpClient()
    val gson = Gson()
    val datasetDir = File(filesDir, "datasets")
    withContext(Dispatchers.IO) {
        localDatasetList
            .forEach { datasetName ->
                try {
                    val url = "https://$BACKEND_URL/dataset/$datasetName"
//                    val url = "http://$BASE_URL:8000/dataset/$datasetName"
                    val request = Request.Builder().url(url).build()
                    val response = client.newCall(request).execute()
                    if (response.isSuccessful) {
                        val body = response.body?.string()
                        val remoteDataset = gson.fromJson<RemoteDataset>(
                            body,
                            object : TypeToken<RemoteDataset>() {}.type
                        )
                        if (remoteDataset != null) {
                            val localDataset = remoteDataset.toDataset()
                            localDataset.absoluteFilePath =
                                datasetDir.resolve(datasetName).absolutePath
                            datasetDataList.add(localDataset)
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
//        .url("http://$BASE_URL:8000/weights/${modelUniqueIdentifier}")
        .url("https://$BACKEND_URL/weights/${modelUniqueIdentifier}")
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
        val modelFile = File(filesDir, "/models/${localModelData.uniqueIdentifier}.ckpt")

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
//            .url("http://$BASE_URL:8000/weights/create")
            .url("https://$BACKEND_URL/weights/create")
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
        val modelFile = File(filesDir, "/models/${modelUniqueIdentifier}.ckpt")

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

        val request = Request.Builder()
//            .url("http://$BASE_URL:8000/weights/${modelUniqueIdentifier}")
            .url("https://$BACKEND_URL/weights/${modelUniqueIdentifier}")
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

suspend fun uploadDataset(
    datasetData: Dataset,
    username: String,
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
        Log.d("uploadDataset", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
//            .url("http://$BASE_URL:8000/dataset/${datasetData.uniqueIdentifier}")
            .url("https://$BACKEND_URL/dataset/${datasetData.uniqueIdentifier}")
            .patch(requestBody)
            .build()

        Log.d("uploadDataset", "Sending PATCH request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "uploadDataset",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("uploadDataset", "Error during model upload: ${e.message}", e)
        e.printStackTrace()
    }
}


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
        Log.d("uploadDatasetNode", "Generated JSON Payload: $payload")

        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
//            .url("http://$BASE_URL:8000/dataset/create")
            .url("https://$BACKEND_URL/dataset/create")
            .post(requestBody)
            .build()

        Log.d("uploadDatasetNode", "Sending POST request to serverUrl")
        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()
            Log.d(
                "uploadDatasetNode",
                "Response Code: ${response.code}, Response Body: ${response.body?.string()}"
            )
        }
    } catch (e: Exception) {
        Log.e("uploadDatasetNode", "Error during model upload: ${e.message}", e)
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
//            .url("http://$BASE_URL:8000/weights/evaluation")
            .url("https://$BACKEND_URL/weights/evaluation")
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
//            .url("http://$BASE_URL:8000/weights/combine")
            .url("https://$BACKEND_URL/weights/combine")
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

suspend fun getAllFinetunedFrom(weight_id: String): List<Finetune> {
    return try {
        val client = OkHttpClient()
        val gson = Gson()
        val url = "https://$BACKEND_URL/weights/finetuned/rels/$weight_id"
//        val url = "http://$BASE_URL:8000/weights/finetuned/rels/$weight_id"
        val request = Request.Builder().url(url).build()

        Log.d("getAllFinetunedFrom", "Sending GET request to serverUrl")

        withContext(Dispatchers.IO) {
            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val body = response.body?.string()
                if (!body.isNullOrEmpty()) {
                    val jsonObject = gson.fromJson<List<Finetune>>(
                        body,
                        object : TypeToken<List<Finetune>>() {}.type
                    )
                    Log.d("getAllFinetunedFrom", "jsonObject: $jsonObject")
                    return@withContext jsonObject
                }
            }
            Log.e("getAllFinetunedFrom", "Response not successful or empty body")
            emptyList()
        }
    } catch (e: Exception) {
        Log.e("getAllFinetunedFrom", "Error fetching rels: ${e.message}", e)
        emptyList()
    }
}

data class UploadFinetuningRelationshipPayload(
    val old_weight_id: String,
    val new_weight_id: String,
    val finetune: Finetune
)


suspend fun uploadFinetuningRelationship(
    relationshipData: LocalRelationship,
    finetuneData: Finetune
) {
    try {
        val client = OkHttpClient()
        val gson = Gson()
        val payload = gson.toJson(
            UploadFinetuningRelationshipPayload(
                new_weight_id = relationshipData.modelUniqueIdentifier,
                old_weight_id = relationshipData.sourceUniqueIdentifiers,
                finetune = finetuneData
            )
        )
        Log.d("uploadRelationship", "Generated JSON Payload: $payload")
        val requestBody = payload.toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("https://$BACKEND_URL/weights/finetuned")
//            .url("http://$BASE_URL:8000/weights/finetuned")
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
            val destinationPath = "weights/$fileName.ckpt"
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
//                .url("http://$BASE_URL:8000/weights/$id")
                .url("https://$BACKEND_URL/weights/$id")
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

fun getDeviceSpecifications(): HashMap<String, String> {
    return hashMapOf(
        "Manufacturer" to Build.MANUFACTURER,
        "Model" to Build.MODEL,
        "Board" to Build.BOARD,
        "Brand" to Build.BRAND,
        "Device" to Build.DEVICE,
        "Product" to Build.PRODUCT,
        "CPU ABI" to Build.SUPPORTED_ABIS.joinToString(", "),
        "Android Version" to Build.VERSION.RELEASE,
        "API Level" to Build.VERSION.SDK_INT.toString()
    )
}