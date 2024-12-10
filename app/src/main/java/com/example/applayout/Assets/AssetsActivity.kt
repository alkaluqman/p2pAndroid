package com.example.applayout.Assets

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import com.example.applayout.Database.AppDatabase


class AssetsActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val database = AppDatabase.getDatabase(this)
        val viewModel = AssetsViewModel(database)
//        copyAssetsToFilesDir(this, "datasets","datasets")
//        copyAssetsToFilesDir(this,"models","models")
//        seedModelsDatabase(this,database)
//        seedDatasetDatabase(this,database)

        setContent {
            MaterialTheme {
                AssetsScreen(
                    viewModel = viewModel,
                    filesDir = filesDir
                )
            }
        }
    }
}
