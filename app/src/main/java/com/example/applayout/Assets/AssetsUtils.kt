package com.example.applayout.Assets

import android.content.Context
import android.util.Log
import com.example.applayout.Database.AppDatabase
import com.example.applayout.Database.Entities.Dataset
import com.example.applayout.Database.Entities.Model
import com.example.applayout.Models.DatasetMetadata
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.util.zip.ZipInputStream


fun getSubfolders(fileDir: File, parentFolder: String): List<String> {
    return try {
        val parentPath = File(fileDir, parentFolder) // Resolve parent folder path
        if (parentPath.exists() && parentPath.isDirectory) {
            parentPath.listFiles()
                ?.filter { it.isDirectory }
                ?.map { it.name }
                ?: emptyList()
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        emptyList()
    }
}


fun getImageCount(filesDir: File, folderPath: String): Int {
    return try {
        val folder = File(filesDir, folderPath)
        if (folder.exists() && folder.isDirectory) {
            val files = folder.listFiles()
            if (files != null) {
                val imageCount = files.count { file ->
                    val isImage = file.extension.lowercase() in listOf("png", "jpg", "jpeg")
                    isImage
                }
                imageCount
            } else {
                Log.d("ImageCountDebug", "No files found in folder.")
                0
            }
        } else {
            Log.d(
                "ImageCountDebug",
                "Folder does not exist or is not a directory: ${folder.absolutePath}"
            )
            0
        }
    } catch (e: Exception) {
        Log.e("ImageCountDebug", "Error while counting images: ${e.message}", e)
        0
    }
}

fun readDatasetMetadata(filePath: String, filesDir: File): DatasetMetadata? {
    return try {
        val metadataFile = File(filesDir, "$filePath/metadata.json")
        if (metadataFile.exists()) {
            val reader = metadataFile.bufferedReader()
            val metadata = Gson().fromJson(reader, DatasetMetadata::class.java)
            reader.close()
            metadata
        } else {
            null // File does not exist
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}


fun getFileSize(filesDir: File, filePath: String): Long {
    return try {
        val file = File(filesDir, filePath)
        Log.d("FileCheck", "Full Path: ${file.absolutePath}")

        if (file.exists()) {
            Log.d("FileCheck", "File exists: ${file.absolutePath}, Size: ${file.length()} bytes")
        } else {
            Log.d("FileCheck", "File does NOT exist: ${file.absolutePath}")
        }

        if (file.exists() && file.isFile) {
            file.length()
        } else {
            0L
        }
    } catch (e: Exception) {
        Log.e("FileError", "Exception: ${e.message}")
        0L
    }
}

fun copyAssetsToFilesDir(context: Context, assetFolder: String, destinationFolder: String) {
    val assetManager = context.assets
    val destinationDir = File(context.filesDir, destinationFolder)

    try {
        val assets = assetManager.list(assetFolder) ?: return
        if (!destinationDir.exists()) {
            destinationDir.mkdirs()
        }
        for (assetName in assets) {
            val assetPath = "$assetFolder/$assetName"
            val destinationFile = File(destinationDir, assetName)
            if (assetManager.list(assetPath)?.isNotEmpty() == true) {
                copyAssetsToFilesDir(context, assetPath, "$destinationFolder/$assetName")
            } else {
                // Otherwise, copy the file
                assetManager.open(assetPath).use { inputStream ->
                    destinationFile.outputStream().use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
        }
    } catch (e: IOException) {
        e.printStackTrace()
    }
}


fun seedModelsDatabase(context: Context, database: AppDatabase) {
    val assetManager = context.assets
    CoroutineScope(Dispatchers.IO).launch {
        try {
            val assets = assetManager.list("models") ?: return@launch
            for (assetName in assets) {
                val metadataPath = "models/$assetName/metadata.json"
                val metadata = try {
                    val inputStream = assetManager.open(metadataPath)
                    val reader = inputStream.bufferedReader()
                    val metadata = Gson().fromJson(reader, Model::class.java)
                    reader.close()
                    metadata
                } catch (e: IOException) {
                    e.printStackTrace()
                    null
                }
                if (metadata != null) {
                    database.modelDao().insertModel(metadata)
                }
            }
            logDatabaseContents(database)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}

fun seedDatasetDatabase(context: Context, database: AppDatabase) {
    val assetManager = context.assets
    CoroutineScope(Dispatchers.IO).launch {
        try {
            val assets = assetManager.list("datasets") ?: return@launch
            for (assetName in assets) {
                val metadataPath = "datasets/$assetName/metadata.json"
                val metadata = try {
                    val inputStream = assetManager.open(metadataPath)
                    val reader = inputStream.bufferedReader()
                    val metadata = Gson().fromJson(reader, Dataset::class.java)
                    reader.close()
                    metadata
                } catch (e: IOException) {
                    e.printStackTrace()
                    null
                }
                if (metadata != null) {
                    database.datasetDao().insertDataset(metadata)
                }
            }
            logDatabaseContents(database)
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}

fun logDatabaseContents(database: AppDatabase) {
    val tag = "DatabaseContents"
    database.modelDao().getAllModels().forEach { model ->
        Log.d(tag, "Model ID: ${model.modelId}")
        Log.d(tag, "Model Name: ${model.name}")
        Log.d(tag, "FILE Path: ${model.filePath}")
        Log.d(tag, "Sync Status: ${model.syncStatus}")
        Log.d(tag, "Last Modified: ${model.lastModified}")
    }
    CoroutineScope(Dispatchers.IO).launch {
        database.datasetDao().getAllDataset().collect { datasetList ->
            datasetList.forEach { dataset ->
                Log.d(tag, "Dataset ID: ${dataset.datasetId}")
                Log.d(tag, "Dataset Name: ${dataset.name}")
                Log.d(tag, "DIR Path: ${dataset.dirPath}")
                Log.d(tag, "Last Modified: ${dataset.lastModified}")
            }
        }
    }
}

fun downloadFile(filesDir: File, relativeFilePath: String, fileUrl: String) {
    val client = OkHttpClient()
    try {
        val targetFile = File(filesDir, relativeFilePath)
        val parentDir = targetFile.parentFile
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs()
        }
        val request = Request.Builder()
            .url(fileUrl)
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw Exception("Failed to download file: ${response.message}")
            }

            val inputStream: InputStream? = response.body?.byteStream()
            val outputStream = FileOutputStream(targetFile)
            inputStream?.use { input ->
                outputStream.use { output ->
                    input.copyTo(output)
                }
            }
        }
    } catch (e: Exception) {
        Log.e("ModelSaving", "Error downloading and saving model: ${e.message}", e)

    }
}

fun downloadAndUnzipFile(
    filesDir: File,
    relativeDirPath: String,
    fileUrl: String
) {
    val client = OkHttpClient()

    try {

        val zipFileDir = File(filesDir, relativeDirPath)
        val zipFile = File(zipFileDir, "${System.currentTimeMillis()}_target.zip")
        if (!zipFileDir.exists()) zipFileDir.mkdirs()
        val request = Request.Builder().url(fileUrl).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw Exception("Failed to download file: ${response.message}")

            response.body?.byteStream()?.use { inputStream ->
                FileOutputStream(zipFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }

        // Unzip the file into the same directory
        unzipFile(zipFile, zipFileDir)
        if (zipFile.exists()) zipFile.delete()

    } catch (e: Exception) {
        Log.e("DownloadFile", "Error downloading and unzipping file: ${e.message}", e)
    }

}

fun unzipFile(zipFile: File, targetDir: File) {
    ZipInputStream(FileInputStream(zipFile)).use { zipInputStream ->
        var zipEntry = zipInputStream.nextEntry
        if (zipEntry == null) {
            Log.e("UnzipFile", "The ZIP file is empty!")
            return
        }

        while (zipEntry != null) {
            val extractedFile = File(targetDir, zipEntry.name)
            // Prevent path traversal attacks
            if (!extractedFile.canonicalPath.startsWith(targetDir.canonicalPath)) {
                throw SecurityException("Invalid zip entry: ${zipEntry.name}")
            }
            if (zipEntry.isDirectory) {
                extractedFile.mkdirs()
            } else {
                extractedFile.parentFile?.let { parent ->
                    if (!parent.exists()) parent.mkdirs()
                }
                FileOutputStream(extractedFile).use { outputStream ->
                    zipInputStream.copyTo(outputStream)
                }
            }

            zipInputStream.closeEntry()
            zipEntry = zipInputStream.nextEntry
        }
    }
}
