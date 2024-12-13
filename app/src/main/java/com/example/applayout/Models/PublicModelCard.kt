package com.example.applayout.Models

import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.applayout.Data.Model.Model
import com.example.applayout.Marketplace.downloadModelFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun PublicModelCard(filesDir: File, modelData: Model, navController: NavController) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
            .clickable {
                val encodedUrl =
                    Uri.encode("http://10.0.2.2:3000/weight/${modelData.uniqueIdentifier}")
                navController.navigate("webview/$encodedUrl")
            },
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = "Model: ${modelData.uniqueIdentifier}",
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "File Size: ${modelData.weight_size} bytes",
                fontSize = 16.sp
            )
            Spacer(modifier = Modifier.height(8.dp))
            IconButton(onClick = {
                CoroutineScope(Dispatchers.IO).launch {
                    downloadModelFile(modelData.public_link, filesDir, modelData.uniqueIdentifier)
                }
            }) {
                Icon(
                    imageVector = Icons.Default.Star,
                    contentDescription = "Download Model",
                )
            }
        }
    }
}
