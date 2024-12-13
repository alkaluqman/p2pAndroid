package com.example.applayout.Marketplace

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.applayout.Data.Model.Model
import com.example.applayout.Models.PublicModelCard
import kotlinx.coroutines.launch
import java.io.File


@Composable
fun MarketplaceScreen(filesDir: File, navController: NavController) {
    val scope = rememberCoroutineScope()
    var modelList by remember { mutableStateOf<List<Model>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    LaunchedEffect(Unit) {
        scope.launch {
            modelList = getMarketplaceFiles()
            isLoading = false
        }
    }
    if (isLoading) {
        CircularProgressIndicator()
    } else {
        Text(
            text = "Marketplace Models",
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 8.dp)
        ) {
            items(modelList) { model ->
                PublicModelCard(filesDir, model, navController)
            }
        }
    }
}