package com.example.applayout.LocalAssets


import android.os.Bundle
import android.util.Log
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
import androidx.compose.material3.CircularProgressIndicator
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
import com.example.applayout.Data.Database.AppDatabase
import com.example.applayout.Data.Model.LocalModel
import com.example.applayout.Data.Model.LocalRelationship
import com.example.applayout.Data.Model.Model
import com.example.applayout.Marketplace.MarketplaceScreen
import com.example.applayout.Marketplace.WebViewScreen
import com.example.applayout.Models.InstalledModelCard
import com.example.applayout.Models.LocalModelCard
import com.example.applayout.Models.LocalRelationshipCard
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
            }
        }
    }
}


val USERNAME = "alice"

@Composable
fun LocalAssetsScreen(filesDir: File, navController: NavController) {
    val uploadedModelListState = remember { mutableStateOf<List<Model>>(emptyList()) }
    val localModelListState = remember { mutableStateOf<List<String>>(emptyList()) }
    val localRelationshipListState =
        remember { mutableStateOf<List<LocalRelationship>>(emptyList()) }
    var showEditLocalDialog by remember { mutableStateOf(false) }
    var showEditInstallDialog by remember { mutableStateOf(false) }
    var showRelationshipDialog by remember { mutableStateOf(false) }
    var selectedInstalledModel by remember { mutableStateOf<Model?>(null) }
    var selectedLocalModel by remember { mutableStateOf<String?>(null) }
    var selectedLocalModelData by remember { mutableStateOf<LocalModel?>(null) }
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    LaunchedEffect(Unit) {
        coroutineScope.launch {
            val results =
                fetchModelsInfo(filesDir) //check if downloaded models are public TODO change logic to storing flag on database
            var updatedModels = results.uploadedModels.toMutableList()
            updatedModels = updatedModels.map { model ->
                model.apply {
                    isOwner = fetchModelOwner(
                        uniqueIdentifier,
                        USERNAME
                    ) //check isOwner for each downloaded model
                }
            }.toMutableList()

            withContext(Dispatchers.Main) { // Switch back to Main thread for UI updates
                uploadedModelListState.value = updatedModels
                localModelListState.value = results.notUploadedModels
            }

            //fetch local relationships
            val db = AppDatabase.getDatabase(context)
            val relationshipDao = db.localRelationshipDao()
            val relationships = relationshipDao.getAll().toMutableList()
            withContext(Dispatchers.Main) { // Update UI state on the Main thread
                localRelationshipListState.value = relationships.toMutableList()
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
            items(uploadedModelListState.value) { model ->
                InstalledModelCard(
                    modelData = model,
                    navController = navController,
                    onClick = { clickedModel ->
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
                    }
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
                text = "Local Models",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    coroutineScope.launch {
                        downloadFile(context, filesDir, "models")
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
                LocalModelCard(
                    model,
                    File(filesDir, "/models/$model.tflite").length(),
                    onEdit = { clickedFileName ->
                        Log.d("LocalAssetsScreen", "onEdit clicked with fileName: $clickedFileName")
                        selectedLocalModel = clickedFileName
                        showEditLocalDialog = true
                        Log.d(
                            "LocalAssetsScreen",
                            "selectedLocalModel: $selectedLocalModel, showEditLocalDialog: $showEditLocalDialog"
                        )
                    },
                    onSend = { clickedFileName ->
                        val db = AppDatabase.getDatabase(context)
                        val modelDao = db.localModelDao()
                        coroutineScope.launch(Dispatchers.IO) {
                            val localModelData = modelDao.getModel(clickedFileName)


                            uploadModel(
                                filesDir,
                                localModelData!!,
                                clickedFileName,
                                "alice",
                                context
                            )
                            modelDao.deleteModel(clickedFileName)
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
                        val db = AppDatabase.getDatabase(context)
                        val modelDao = db.localModelDao()
                        coroutineScope.launch(Dispatchers.IO) {
                            modelDao.deleteModel(filename)
                            localModelListState.value = fetchModelsInfo(filesDir).notUploadedModels
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
            Text(
                text = "Local Relationships",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f)
            )
            Button(
                onClick = {
                    showRelationshipDialog = true
                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Create Relationship")
            }
        }
        LazyColumn(
            modifier = Modifier
                .padding(top = 8.dp)
                .height(250.dp)
        ) {
            items(localRelationshipListState.value) { localRelationship ->
                LocalRelationshipCard(
                    relationship = localRelationship,
                    onSend = { relationshipData ->
                        val db = AppDatabase.getDatabase(context)
                        val relationshipDao = db.localRelationshipDao()
                        coroutineScope.launch(Dispatchers.IO) {
                            relationshipDao.deleteById(relationshipData.modelUniqueIdentifier)
                            uploadRelationship(
                                relationshipData
                            )
                            localRelationshipListState.value =
                                relationshipDao.getAll().toMutableList()
                        }
                    },
                    onDelete = { relationshipID ->
                        val db = AppDatabase.getDatabase(context)
                        val relationshipDao = db.localRelationshipDao()
                        coroutineScope.launch(Dispatchers.IO) {
                            relationshipDao.deleteById(relationshipID)
                            localRelationshipListState.value =
                                relationshipDao.getAll().toMutableList()
                        }
                    }
                )
            }
        }
    }
    if (showEditLocalDialog && selectedLocalModel != null) {
        LaunchedEffect(selectedLocalModel) {
            val db = AppDatabase.getDatabase(context)
            val modelDao = db.localModelDao()
            try {
                selectedLocalModelData = withContext(Dispatchers.IO) {
                    modelDao.getModel(selectedLocalModel!!)
                }
            } catch (e: Exception) {
                showEditLocalDialog = false
            }
        }

        if (selectedLocalModelData != null) {
            EditModelDialog(
                onDismiss = {
                    showEditLocalDialog = false
                    selectedLocalModelData = null // Reset state after dismissing
                },
                localModelData = selectedLocalModelData!!,
                onSubmit = { formData ->
                    coroutineScope.launch(Dispatchers.IO) {
                        val db = AppDatabase.getDatabase(context)
                        val modelDao = db.localModelDao()
                        try {
                            modelDao.updateModel(
                                selectedLocalModelData!!.uniqueIdentifier,
                                formData.description,
                                formData.model_task

                            )
                            withContext(Dispatchers.Main) {
                                showEditLocalDialog = false
                                selectedLocalModelData = null // Reset state after saving
                            }
                        } catch (e: Exception) {
                            Log.e("LocalAssetsScreen", "Error updating data: ${e.message}")
                        }
                    }
                }
            )
        } else {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        }
    }
    if (showEditInstallDialog && selectedInstalledModel != null) {
        EditModelDialog(
            onDismiss = { showEditInstallDialog = false },
            localModelData = selectedInstalledModel!!.toLocalModel(),
            onSubmit = { formData ->
                coroutineScope.launch {
                    editModel(selectedInstalledModel!!.uniqueIdentifier, formData)
                    val results = fetchModelsInfo(filesDir)
                    uploadedModelListState.value = results.uploadedModels
                }
                showEditInstallDialog = false

            }
        )
    }
    if (showRelationshipDialog) {
        EditRelationshipDialog(
            onDismiss = { showRelationshipDialog = false },
            uploadedModels = uploadedModelListState.value.map { it.uniqueIdentifier },
            localModels = localModelListState.value,
            onSubmit = { localRelationship ->
                coroutineScope.launch(Dispatchers.IO) {
                    val db = AppDatabase.getDatabase(context)
                    val relationshipDao = db.localRelationshipDao()
                    try {
                        relationshipDao.insert(localRelationship)
                        localRelationshipListState.value = relationshipDao.getAll().toMutableList()
                    } catch (e: Exception) {
                        Log.e("LocalAssetsScreen", "Error updating relationship: ${e.message}")
                    }
                }
                showRelationshipDialog = false
            }
        )
    }
}

