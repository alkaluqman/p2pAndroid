package com.example.applayout.LocalAssets

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

@Composable
fun EditDatasetDialog(
    datasetName: String,
    filesDir: File,
    onDismiss: () -> Unit,
    onSubmit: () -> Unit
) {
//    var datasetData by remember { mutableStateOf(localDatasetData) }
    var fileNames by remember {
        mutableStateOf(
            listLocalResources(filesDir, "datasets/${datasetName}", true)
                .filter { it != "labels.json" }
        )
    }
    val datasetDir = File(filesDir, "datasets/${datasetName}")
    var datasetLabels = readLabels(datasetDir).toMutableMap()
    val context = LocalContext.current
    val launcher =
        rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                saveImageToDataset(context, it, datasetDir)?.let { newFileName ->
                    fileNames = listLocalResources(filesDir, "datasets/${datasetName}", true)
                        .filter { it != "labels.json" }
                }
            }
        }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(dismissOnBackPress = true, dismissOnClickOutside = true)
    ) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.padding(16.dp),
            tonalElevation = 4.dp
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Edit Dataset",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                LazyColumn(modifier = Modifier.fillMaxHeight(0.6f)) {
                    items(fileNames) { filename ->
                        DatasetItemCard(
                            datasetDir = datasetDir,
                            fileName = filename,
                            label = datasetLabels[filename] ?: "na",
                            onLabelChange = { newLabel ->
                                datasetLabels[filename] = newLabel
                                val labelsFile = File(datasetDir, "labels.json")
                                labelsFile.writeText(Gson().toJson(datasetLabels))
                                datasetLabels = readLabels(datasetDir).toMutableMap()
                            },
                            onFileNameChange = { newFileName ->
                                val oldFile = File(datasetDir, filename)
                                val newFile = File(datasetDir, newFileName)
                                if (oldFile.renameTo(newFile)) {
                                    fileNames =
                                        fileNames.map { if (it == filename) newFileName else it }
                                }
                            },
                            onDelete = {
                                val file = File(datasetDir, filename)
                                if (file.delete()) {
                                    // Remove the entry from datasetLabels
                                    datasetLabels.remove(filename)
                                    val labelsFile = File(datasetDir, "labels.json")
                                    labelsFile.writeText(Gson().toJson(datasetLabels))
                                    fileNames = fileNames.filter { it != filename }
                                }
                            }
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Button(
                    onClick = {
                        launcher.launch("image/*")
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Upload New File")
                }
            }
        }
    }
}

@Composable
fun DatasetItemCard(
    datasetDir: File,
    fileName: String,
    label: String,
    onLabelChange: (String) -> Unit,
    onFileNameChange: (String) -> Unit,
    onDelete: () -> Unit
) {
    val imageFilePath = File(datasetDir, fileName).absolutePath
    var newFileName by remember { mutableStateOf(fileName) }
    var newLabel by remember { mutableStateOf(label) }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
            .border(1.dp, MaterialTheme.colorScheme.primary),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Filename:")
                Spacer(modifier = Modifier.width(8.dp))
                BasicTextField(
                    value = newFileName,
                    onValueChange = { updatedFileName ->
                        newFileName = updatedFileName
                        onFileNameChange(updatedFileName)
                    },
                    modifier = Modifier
                        .border(1.dp, MaterialTheme.colorScheme.primary)
                        .padding(8.dp)
                        .fillMaxWidth(0.7f) // Adjust width as needed
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Label:")
                Spacer(modifier = Modifier.width(8.dp))
                BasicTextField(
                    value = newLabel,
                    onValueChange = { updatedLabel ->
                        newLabel = updatedLabel // Update the local state
                        onLabelChange(updatedLabel) // Pass the updated label to the callback
                    },
                    modifier = Modifier
                        .border(1.dp, MaterialTheme.colorScheme.primary)
                        .padding(8.dp)
                )
            }
        }
        Image(
            bitmap = loadImageBitmap(imageFilePath),
            contentDescription = null,
            modifier = Modifier.size(64.dp)
        )
        IconButton(onClick = onDelete) {
            Icon(imageVector = Icons.Default.Delete, contentDescription = "Delete")
        }
    }
}


fun loadImageBitmap(filePath: String): androidx.compose.ui.graphics.ImageBitmap {
    val bitmap = BitmapFactory.decodeFile(filePath)
    return bitmap?.asImageBitmap() ?: androidx.compose.ui.graphics.ImageBitmap(1, 1)
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


fun saveImageToDataset(context: Context, uri: Uri, datasetDir: File): String? {
    return try {
        val inputStream: InputStream? = context.contentResolver.openInputStream(uri)
        val originalFileName = uri.lastPathSegment?.substringAfterLast("/") ?: "new_image"
        val fileExtension =
            context.contentResolver.getType(uri)?.substringAfter("/") ?: "png" //default
        val fileName =
            if (originalFileName.contains(".")) originalFileName else "$originalFileName.$fileExtension"
        val outputFile = File(datasetDir, fileName)
        inputStream?.use { input ->
            FileOutputStream(outputFile).use { output ->
                input.copyTo(output)
            }
        }
        fileName
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}