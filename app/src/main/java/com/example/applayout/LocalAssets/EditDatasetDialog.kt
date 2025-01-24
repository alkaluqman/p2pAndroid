package com.example.applayout.LocalAssets

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Data.Model.modelTasks
import com.example.applayout.Models.loadImageBitmap
import com.example.applayout.Models.readLabels
import com.google.gson.Gson
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream


@Composable
fun EditDatasetDialog(
    datasetData: Dataset,
    filesDir: File,
    onDismiss: () -> Unit,
    onSubmit: (Dataset) -> Unit
) {

    var modelTask by remember { mutableStateOf(datasetData.model_task) }
    var description by remember { mutableStateOf(datasetData.description) }
    Log.d("update dataset", "Dataset ID1: $datasetData")
    var fileNames by remember {
        mutableStateOf(
            listLocalResources(filesDir, "datasets/${datasetData.uniqueIdentifier}", true)
                .filter { it != "labels.json" }
        )
    }
    val datasetDir = File(filesDir, "datasets/${datasetData.uniqueIdentifier}")
    var datasetLabels = readLabels(datasetDir).toMutableMap()
    Log.d("update dataset", "Dataset ID2: $datasetData")
    val context = LocalContext.current
    val launcher =
        rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                saveImageToDataset(context, it, datasetDir)?.let { newFileName ->
                    fileNames = listLocalResources(
                        filesDir,
                        "datasets/${datasetData.uniqueIdentifier}",
                        true
                    )
                        .filter { it != "labels.json" }
                }
            }
        }
    Log.d("update dataset", "Dataset ID3: $datasetData")
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
                Text("Model Task")
                var expanded by remember { mutableStateOf(false) }
                Box {
                    Text(
                        text = modelTask,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { expanded = true }
                            .border(1.dp, MaterialTheme.colorScheme.primary)
                    )
                    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        modelTasks.forEach { task ->
                            DropdownMenuItem(
                                text = { Text(task) },
                                onClick = {
                                    modelTask = task
                                    expanded = false
                                }
                            )
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text("Description")
                BasicTextField(
                    value = description,
                    onValueChange = { description = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .border(1.dp, MaterialTheme.colorScheme.primary)
                        .padding(8.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))

                LazyColumn(modifier = Modifier.fillMaxHeight(0.6f)) {
                    items(fileNames) { filename ->
                        DatasetItemCard(
                            classLabels = datasetData.getClassLabelsAsList(),
                            datasetDir = datasetDir,
                            fileName = filename,
                            label = datasetLabels[filename] ?: "0",
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

                Button(
                    onClick = {
                        onSubmit(
                            Dataset(
                                uniqueIdentifier = datasetData.uniqueIdentifier,
                                model_task = modelTask,
                                description = description,
                            )
                        )
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Save Changes")
                }
            }
        }
    }
}

@Composable
fun DatasetItemCard(
    classLabels: List<String>,
    datasetDir: File,
    fileName: String,
    label: String,
    onLabelChange: (String) -> Unit,
    onFileNameChange: (String) -> Unit,
    onDelete: () -> Unit
) {
    val imageFilePath = File(datasetDir, fileName).absolutePath
    var newFileName by remember { mutableStateOf(fileName) }
    var newLabel by remember { mutableStateOf(label.toIntOrNull() ?: 0) }
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
            Row() {
                Text("Label:")
                var expanded by remember { mutableStateOf(false) }

                Box(modifier = Modifier.fillMaxWidth()) {
                    Button(
                        onClick = { expanded = true },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Select Label: $newLabel")
                    }

                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        classLabels.forEachIndexed { index, classLabel ->
                            DropdownMenuItem(
                                text = { Text("$index - $classLabel") }, // Display number and item
                                onClick = {
                                    newLabel = index
                                    onLabelChange(newLabel.toString())
                                    expanded = false
                                }
                            )
                        }
                    }
                }
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