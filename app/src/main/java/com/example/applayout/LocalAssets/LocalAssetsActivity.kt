package com.example.applayout.LocalAssets


import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.applayout.Data.Model.Dataset
import com.example.applayout.Data.Model.Finetune
import com.example.applayout.Data.Model.Model
import com.example.applayout.Dataset.DatasetCard
import com.example.applayout.Marketplace.MarketplaceScreen
import com.example.applayout.Marketplace.WebViewScreen
import com.example.applayout.Models.ModelCard
import com.example.applayout.Models.runInferenceOnDirectory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

import com.example.applayout.Finetune.FinetuneAPI
import com.example.applayout.Metrics.MetricTracking
import com.google.gson.Gson

class LocalAssetActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val navController = rememberNavController()
            NavHost(navController, startDestination = "local_assets") {
                composable("local_assets") { LocalAssetsScreen(filesDir, navController) }
                composable("marketplace") { MarketplaceScreen(filesDir, navController) }
                composable(
                    "webview/{url}",
                    arguments = listOf(navArgument("url") { type = NavType.StringType })
                ) { backStackEntry ->
                    val url = backStackEntry.arguments?.getString("url") ?: "https://www.google.com"
                    WebViewScreen(url = url)
                }
            }
        }
    }
}


val USERNAME = "alice"

@Composable
fun LocalAssetsScreen(filesDir: File, navController: NavController) {
    val uploadedModelListState = remember { mutableStateOf<List<Model>>(emptyList()) }
    val localModelListState = remember { mutableStateOf<List<Model>>(emptyList()) }
    val localDatasetListState = remember { mutableStateOf<List<Dataset>>(emptyList()) }
    var showEditInstallDialog by remember { mutableStateOf(false) }
    var showEditDatabaseDialog by remember { mutableStateOf(false) }
    var showFederatedLearningRelationshipDialog by remember { mutableStateOf(false) }
    var showFinetuningRelationshipDialog by remember { mutableStateOf(false) }
    var selectedInstalledModel by remember { mutableStateOf<Model?>(null) }
    var selectedLocalDataset by remember { mutableStateOf<Dataset?>(null) }
    var showResultsDialog by remember { mutableStateOf(false) }
    var highlightedModel by remember { mutableStateOf<String?>(null) }
    val context = LocalContext.current
    val internetConnected = remember { mutableStateOf(false) }
    var evaluationResults by remember { mutableStateOf<List<Pair<Int, Int>>>(emptyList()) }
    var evaluationClasses by remember { mutableStateOf<List<String>>(emptyList()) }
    var evaluatedModelId by remember { mutableStateOf<String?>(null) }
    var evaluatedDatasetId by remember { mutableStateOf<String?>(null) }
    var evaluatedDatasetUploadStatus by remember { mutableStateOf<Boolean>(false) }
    val coroutineScope = rememberCoroutineScope()
    LaunchedEffect(Unit) {
        coroutineScope.launch {
            internetConnected.value = isInternetConnected(context)
            val results =
                fetchModelsInfo(filesDir)
            var updatedModels = results.uploadedModels.toMutableList()
            updatedModels = updatedModels.map { model ->
                model.apply {
                    isOwner = fetchModelOwner(
                        uniqueIdentifier,
                        USERNAME
                    )
                }
            }.toMutableList()
            val localDatasetNameList = listLocalResources(filesDir, "datasets", false)

            withContext(Dispatchers.Main) { // Switch back to Main thread for UI updates
                uploadedModelListState.value = updatedModels
                localModelListState.value = results.notUploadedModels
                localDatasetListState.value = fetchDatasetInfo(filesDir, localDatasetNameList)
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.primary)
                .padding(16.dp)
        ) {
            Text(
                text = "Local Assets",
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.align(Alignment.Center)
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Public Models",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = { navController.navigate("marketplace") },
                enabled = internetConnected.value,
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Go to Marketplace")
            }
        }
        LazyColumn(
            modifier = Modifier
                .padding(top = 8.dp)
                .height(250.dp)
        ) {
            if (!internetConnected.value) {
                item {
                    Text(
                        text = "Internet connection is required to access the Marketplace.",
                        color = MaterialTheme.colorScheme.error,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }
            } else {
                items(uploadedModelListState.value) { model ->
                    ModelCard(
                        isSelected = highlightedModel == model.uniqueIdentifier,
                        modelData = model,
                        navController = navController,
                        onClick = { clickedModelId ->
                            highlightedModel = clickedModelId
                        },
                        onEdit = { clickedModel ->
                            selectedInstalledModel = clickedModel
                            showEditInstallDialog = true
                        },
                        onDelete = { filename ->
                            deleteLocalFile(filename, filesDir, "models")
                            coroutineScope.launch(Dispatchers.IO) {
                                val results = fetchModelsInfo(filesDir)
                                var updatedModels = results.uploadedModels.toMutableList()
                                updatedModels = updatedModels.map { model ->
                                    model.apply {
                                        isOwner = fetchModelOwner(uniqueIdentifier, USERNAME)
                                    }
                                }.toMutableList()
                                uploadedModelListState.value = updatedModels
                            }
                        },
                        onUpload = {},
                        isUploaded = true

                    )
                }
            }
        }


        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Local Models",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    coroutineScope.launch {
                        val newModel = downloadFile(filesDir, "models")
                        uploadModelNode(
                            filesDir,
                            newModel!!,
                            "alice",
                        )
                        localModelListState.value = fetchModelsInfo(filesDir).notUploadedModels
                    }
                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Spawn Local Files")
            }
        }
        LazyColumn(
            modifier = Modifier
                .padding(top = 8.dp)
                .height(250.dp)
        ) {
            items(localModelListState.value) { model ->
                ModelCard(
                    isSelected = highlightedModel == model.uniqueIdentifier,
                    model,
                    navController = navController,
                    onClick = { clickedModelId ->
                        highlightedModel = clickedModelId
                    },
                    onEdit = { clickedModel ->
                        selectedInstalledModel = clickedModel
                        showEditInstallDialog = true
                    },
                    onUpload = { clickedFileName ->
                        coroutineScope.launch(Dispatchers.IO) {
                            uploadModel(
                                filesDir,
                                clickedFileName,
                                context
                            )
                            val results = fetchModelsInfo(filesDir)
                            var updatedModels = results.uploadedModels.toMutableList()
                            updatedModels = updatedModels.map { model ->
                                model.apply {
                                    isOwner = fetchModelOwner(uniqueIdentifier, USERNAME)
                                }
                            }.toMutableList()
                            uploadedModelListState.value = updatedModels
                            localModelListState.value = results.notUploadedModels
                        }
                    },
                    onDelete = { filename ->
                        deleteLocalFile(filename, filesDir, "models")
                        coroutineScope.launch(Dispatchers.IO) {
                            localModelListState.value = fetchModelsInfo(filesDir).notUploadedModels
                        }
                    },
                    isUploaded = false
                )
            }
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),

            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Local Datasets",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    coroutineScope.launch {
                        createDatasetFolder(filesDir, USERNAME)
                        val localDatasetNameList = listLocalResources(filesDir, "datasets", false)
                        withContext(Dispatchers.Main) {
                            localDatasetListState.value =
                                fetchDatasetInfo(filesDir, localDatasetNameList)
                        }
                    }
                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text(" Add Dataset Folder")
            }
        }
        LazyColumn(
            modifier = Modifier
                .padding(top = 8.dp)
                .height(250.dp)
        ) {
            items(localDatasetListState.value) { dataset ->
                DatasetCard(
                    dataset,
                    numImages = countFilesInDirectory(
                        filesDir,
                        "datasets/${dataset.uniqueIdentifier}"
                    ) - 1,
                    onRemove = { selectedDatasetName ->
                        deleteLocalDirectory(selectedDatasetName, filesDir)
                        coroutineScope.launch(Dispatchers.IO) {
                            val localDatasetNameList =
                                listLocalResources(filesDir, "datasets", false)
                            withContext(Dispatchers.Main) {
                                localDatasetListState.value =
                                    fetchDatasetInfo(filesDir, localDatasetNameList)
                            }
                        }
                    },
                    onEdit = { selectedDataset ->
                        selectedLocalDataset = selectedDataset
                        showEditDatabaseDialog = true
                        Log.d(
                            "LocalAssetsScreen",
                            "selectedLocalDataset: $selectedLocalDataset, showEditDatabaseDialog: $showEditDatabaseDialog"
                        )
                    },
                    onRun = { selectedDataset ->
                        if (highlightedModel != null) {
                            evaluationResults = runInferenceOnDirectory(
                                context,
                                modelId = highlightedModel!!,
                                datasetId = selectedDataset.uniqueIdentifier
                            )
                            coroutineScope.launch(Dispatchers.IO) {
                                evaluationClasses = selectedDataset.getClassLabelsAsList()
                                evaluatedDatasetId = selectedDataset.uniqueIdentifier
                                evaluatedModelId = highlightedModel
                                evaluatedDatasetUploadStatus = selectedDataset.isUploaded
                            }
                            showResultsDialog = true
                        } else {
                            Toast.makeText(
                                context,
                                "Please select a model before running inference!",
                                Toast.LENGTH_SHORT
                            ).show()
                        }


                    })

            }

        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),

            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(
                onClick = {
                    showFederatedLearningRelationshipDialog = true

                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Federated Learn")
            }
            Button(
                onClick = {
                    showFinetuningRelationshipDialog = true

                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Finetune")
            }
        }
        if (showEditInstallDialog && selectedInstalledModel != null) {
            EditModelDialog(
                onDismiss = { showEditInstallDialog = false },
                modelData = selectedInstalledModel!!,
                onSubmit = { modelData ->
                    coroutineScope.launch {
                        editModel(selectedInstalledModel!!.uniqueIdentifier, modelData)
                        val results = fetchModelsInfo(filesDir)
                        var updatedModels = results.uploadedModels.toMutableList()
                        updatedModels = updatedModels.map { model ->
                            model.apply {
                                isOwner = fetchModelOwner(
                                    uniqueIdentifier,
                                    USERNAME
                                )
                            }
                        }.toMutableList()
                        uploadedModelListState.value = updatedModels
                    }
                    showEditInstallDialog = false

                }
            )
        }
        if (showFederatedLearningRelationshipDialog) {
            EditFederatedLearningRelationshipDialog(
                onDismiss = { showFederatedLearningRelationshipDialog = false },
                models = uploadedModelListState.value + localModelListState.value,
                onSubmit = { localRelationship ->
                    coroutineScope.launch(Dispatchers.IO) {
                        uploadFederatedLearningRelationship(localRelationship)
                    }
                    showFederatedLearningRelationshipDialog = false
                }
            )
        }
        if (showFinetuningRelationshipDialog) {
            EditFinetuningRelationshipDialog(
                onDismiss = { showFinetuningRelationshipDialog = false },
                models = uploadedModelListState.value + localModelListState.value,
                dataset = localDatasetListState.value,
                onSubmit = { localRelationship, modelFileName, datasetDirName, numEpochs, batchSize ->
                    coroutineScope.launch(Dispatchers.IO) {
                        val trackingResults: HashMap<String, Double> =
                            MetricTracking.doWithTracking {
                                FinetuneAPI.finetune(
                                    context,
                                    filesDir,
                                    modelFileName,
                                    datasetDirName,
                                    numEpochs,
                                    batchSize
                                )
                            }
                        Log.d("LocalAssetsScreen", "Tracking Results: $trackingResults")

                        val deviceSpecifications: HashMap<String, String> =
                            getDeviceSpecifications()
                        Log.d("LocalAssetsScreen", "Device Specs: $deviceSpecifications")

                        val mergedMap = trackingResults + deviceSpecifications
                        val performanceJson = Gson().toJson(mergedMap)
                        Log.d("LocalAssetsScreen", "Performance Json: $performanceJson")

                        val finetune = Finetune(
                            num_epochs = numEpochs,
                            batch_size = batchSize,
                            performance_json = performanceJson
                        )
                        uploadFinetuningRelationship(
                            relationshipData = localRelationship,
                            finetuneData = finetune
                        )
                    }
                    showFinetuningRelationshipDialog = false
                }
            )
        }

        if (showEditDatabaseDialog && selectedLocalDataset != null) {
            EditDatasetDialog(
                datasetData = selectedLocalDataset!!,
                filesDir = filesDir,
                onDismiss = { showEditDatabaseDialog = false },
                onSubmit = { newDatasetData ->
                    coroutineScope.launch(Dispatchers.IO) {
                        withContext(Dispatchers.Main) {
                            uploadDataset(newDatasetData, USERNAME, filesDir)
                            showEditDatabaseDialog = false
                            selectedLocalDataset = null
                        }
                    }
                }
            )
        }
    }
    if (showResultsDialog) {
        EvaluationResultsDialog(
            isUploadable = true,
//                isUploadable = evaluatedDatasetUploadStatus && uploadedModelListState.value.any { it.uniqueIdentifier == highlightedModel },
            results = evaluationResults,
            classLabels = evaluationClasses,
            onDismiss = { showResultsDialog = false },
            onUpload = { results ->
                coroutineScope.launch(Dispatchers.IO) {
                    uploadEvaluationResults(evaluatedModelId!!, evaluatedDatasetId!!, results)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(
                            context,
                            "Results uploaded successfully!",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            }
        )
    }

}

