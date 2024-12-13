package com.example.applayout.Marketplace

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.navigation.NavController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.applayout.Data.Model.Model
import kotlinx.coroutines.launch
import java.io.File

class MarketplaceActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val navController = rememberNavController()
            NavHost(navController, startDestination = "home") {
                composable("home") { MarketplaceScreen(filesDir, navController) }
                composable(
                    "webview/{url}",
                    arguments = listOf(navArgument("url") { type = NavType.StringType })
                ) { backStackEntry ->
                    val url = backStackEntry.arguments?.getString("url") ?: "https://www.google.com"
                    WebViewScreen(url = url)
                }
            }
//            MarketplaceScreen(filesDir = filesDir)
        }
    }
}

@Composable
fun MarketplaceScreen(filesDir: File, navController: NavController) {
    val scope = rememberCoroutineScope()
    var modelList by remember { mutableStateOf<List<Model>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    LaunchedEffect(Unit) {
        scope.launch {
            modelList = getMarketplaceFiles() // todo filter out existing local models
            isLoading = false
        }
    }

    if (isLoading) {
        CircularProgressIndicator()
    } else {
        if (modelList.isNotEmpty()) {
            Column {
                modelList.forEach { model ->
                    ModelCard(filesDir, model, navController)
                }
            }
        } else {
            Text("No models found or an error occurred.")
        }
    }
}
