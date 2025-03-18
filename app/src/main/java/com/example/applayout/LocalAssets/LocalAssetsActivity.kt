package com.example.applayout.LocalAssets


import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
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
import androidx.compose.runtime.saveable.rememberSaveable
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
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.Data.Model.Model
import com.example.applayout.Dataset.DatasetCard
import com.example.applayout.Finetune.FinetuneAPI
import com.example.applayout.Marketplace.MarketplaceScreen
import com.example.applayout.Marketplace.PerformanceHistoryScreen
import com.example.applayout.Marketplace.LastFinetuneScreen
import com.example.applayout.Marketplace.WebViewScreen
import com.example.applayout.Metrics.MetricTracking
import com.example.applayout.Models.ModelCard
import com.example.applayout.Models.runInferenceOnDirectory
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

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
                composable("performance_history/{listJson}") { backStackEntry ->
                    val json = backStackEntry.arguments?.getString("listJson") ?: "[]"
                    val listType = object : TypeToken<List<Finetune>>() {}.type
                    val list: List<Finetune> = Gson().fromJson(json, listType)

                    PerformanceHistoryScreen(list)
                }
                composable("last_finetune_screen/{lastRelationshipJson}/{finetuneJson}") { backStackEntry ->
                    val lastRelationshipJson =
                        backStackEntry.arguments?.getString("lastRelationshipJson") ?: "{}"
                    val finetuneJson = backStackEntry.arguments?.getString("finetuneJson") ?: "{}"
                    val lastRelationshipJsonType = object : TypeToken<LocalRelationship>() {}.type
                    val finetuneJsonType = object : TypeToken<Finetune>() {}.type

                    val lastRelationship: LocalRelationship =
                        Gson().fromJson(lastRelationshipJson, lastRelationshipJsonType)
                    val finetune: Finetune = Gson().fromJson(finetuneJson, finetuneJsonType)

                    LastFinetuneScreen(
                        context = LocalContext.current,
                        localRelationship = lastRelationship,
                        finetune = finetune
                    )
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
    var showEditDatasetDialog by remember { mutableStateOf(false) }
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
    var lastLocalRelationship by rememberSaveable { mutableStateOf<LocalRelationship?>(null) }
    var lastFinetuneData by rememberSaveable { mutableStateOf<Finetune?>(null) }
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
                text = "Model Manager",
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.titleLarge,
                modifier = Modifier.align(Alignment.Center)
            )
        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(
                onClick = { showFinetuningRelationshipDialog = true },
                modifier = Modifier.weight(1f) // Makes button occupy equal space
            ) {
                Text("Finetune")
            }

            Spacer(modifier = Modifier.width(16.dp))
            Button(
                onClick = { showFederatedLearningRelationshipDialog = true },
                modifier = Modifier.weight(1f) // Makes button occupy equal space
            ) {
                Text("Federated Learn")
            }

            Spacer(modifier = Modifier.width(16.dp))
            Button(

                onClick = {
                    val lastRelationshipJson = Uri.encode(Gson().toJson(lastLocalRelationship))
                    val lastFinetuneJson = Uri.encode(Gson().toJson(lastFinetuneData))
                    navController.navigate("last_finetune_screen/$lastRelationshipJson/$lastFinetuneJson")
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("Last Run Statistics")
            }

        }
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Downloaded Models",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = { navController.navigate("marketplace") },
                enabled = internetConnected.value,
            ) {
                Text("Browse Marketplace")
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
                text = "Personal Models",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    coroutineScope.launch {
//                        val newModel = downloadFile(filesDir, "models")
                        val newModel = getNewModel()
                        FinetuneAPI.getBaseCkptFile(context, filesDir, newModel.uniqueIdentifier)
                        uploadModelNode(
                            filesDir,
                            newModel,
                            "alice",
                        )
                        localModelListState.value = fetchModelsInfo(filesDir).notUploadedModels
                    }
                }
            ) {
                Text("New Model")
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
                    onViewPerformanceHistory = {
                        coroutineScope.launch(Dispatchers.IO) {
                            val allFinetunedRels = getAllFinetunedFrom(model.uniqueIdentifier)
                            Log.d("LocalAssetsActivity", "allFinetunedRels: $allFinetunedRels")

                            val json = Uri.encode(Gson().toJson(allFinetunedRels))
                            withContext(Dispatchers.Main) {
                                navController.navigate("performance_history/$json")
                            }
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
                text = "Personal Datasets",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    coroutineScope.launch {
//                        createDatasetFolder(filesDir, USERNAME)
                        populateDataset(filesDir, USERNAME)
                        val localDatasetNameList = listLocalResources(filesDir, "datasets", false)
                        withContext(Dispatchers.Main) {
                            localDatasetListState.value =
                                fetchDatasetInfo(filesDir, localDatasetNameList)
                        }
                    }
                },
            ) {
                Text(" New Dataset")
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
                        showEditDatasetDialog = true
                        Log.d(
                            "LocalAssetsScreen",
                            "selectedLocalDataset: $selectedLocalDataset, showEditDatabaseDialog: $showEditDatasetDialog"
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
                onSubmit = { sourceModels ->
                    val newModel = getNewModel()
                    val localRelationship = LocalRelationship(
                        modelUniqueIdentifier = newModel.uniqueIdentifier,
                        relationshipType = "FederatedLearn",
                        sourceUniqueIdentifiers = Gson().toJson(sourceModels)
                    )
                    FinetuneAPI.federatedLearn(
                        context,
                        filesDir,
                        sourceModels,
                        newModel.uniqueIdentifier,
                    )
                    coroutineScope.launch(Dispatchers.IO) {
                        uploadModelNode(filesDir, newModel, USERNAME)
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
                onSubmit = { modelFileName, datasetDirName, newModelName, numEpochs, batchSize ->
                    coroutineScope.launch(Dispatchers.IO) {
                        Log.d("LocalAssetsScreen", "modelFileName: $modelFileName")
                        Log.d("LocalAssetsScreen", "datasetDirName: $datasetDirName")
                        Log.d("LocalAssetsScreen", "newModelName: $newModelName")
                        val newModel = getNewModel(newModelName.ifBlank { null })
                        val trackingResults: HashMap<String, Double> =
                            MetricTracking.doWithTracking {
                                FinetuneAPI.finetune(
                                    context,
                                    filesDir,
                                    modelFileName,
                                    datasetDirName,
                                    numEpochs,
                                    batchSize,
                                    newModel.uniqueIdentifier
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
                            performance_json = performanceJson,
                            dataset = datasetDirName
                        )
                        saveFinetune(filesDir, finetune)
                        val localRelationship = LocalRelationship(
                            modelUniqueIdentifier = newModel.uniqueIdentifier,
                            relationshipType = "Finetune",
                            sourceUniqueIdentifiers = modelFileName
                        )
                        lastLocalRelationship = localRelationship
                        lastFinetuneData = finetune
                        uploadModelNode(filesDir, newModel, USERNAME)
                        uploadFinetuningRelationship(
                            relationshipData = localRelationship,
                            finetuneData = finetune
                        )
                        localModelListState.value = fetchModelsInfo(filesDir).notUploadedModels
                    }
                    showFinetuningRelationshipDialog = false
                }
            )
        }

        if (showEditDatasetDialog && selectedLocalDataset != null) {
            EditDatasetDialog(
                datasetData = selectedLocalDataset!!,
                filesDir = filesDir,
                onDismiss = { showEditDatasetDialog = false },
                onSubmit = { newDatasetData ->
                    coroutineScope.launch(Dispatchers.IO) {
                        withContext(Dispatchers.Main) {
                            uploadDataset(newDatasetData, USERNAME, filesDir)
                            showEditDatasetDialog = false
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

