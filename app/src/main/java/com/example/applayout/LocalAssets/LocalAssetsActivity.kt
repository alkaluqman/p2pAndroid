package com.example.applayout.LocalAssets

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.applayout.Data.Model.Model
import com.example.applayout.Marketplace.MarketplaceScreen
import com.example.applayout.Marketplace.WebViewScreen
import com.example.applayout.Models.LocalModelCard
import com.example.applayout.Models.PublicModelCard
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

@Composable
fun LocalAssetsScreen(filesDir: File, navController: NavController) {
    val uploadedModelListState = remember { mutableStateOf<List<Model>>(emptyList()) }
    val localModelListState = remember { mutableStateOf<List<String>>(emptyList()) }
    LaunchedEffect(Unit) {
        val localFiles = getLocalFiles(filesDir, "models")
        val results = fetchModelsInfo(localFiles)
        uploadedModelListState.value = results.uploadedModels
        localModelListState.value = results.notUploadedModels
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
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
                .fillMaxSize()
                .padding(top = 8.dp)
        ) {
            items(uploadedModelListState.value) { model ->
                PublicModelCard(filesDir, model, navController)
            }
        }
        Text(
            text = "Local Models",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.weight(1f)
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 8.dp)
        ) {
            items(localModelListState.value) { model ->
                LocalModelCard(model)
            }
        }
    }
}

