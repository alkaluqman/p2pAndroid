package com.example.applayout.LocalAssets

import java.io.File

fun getLocalFiles(fileDir: File, parentFolder: String): List<String> {
    return try {
        val modelsDir = File(fileDir, parentFolder)
        if (modelsDir.exists() && modelsDir.isDirectory) {
            modelsDir.listFiles()
                ?.filter { it.isFile }
                ?.map { it.name }
                ?: emptyList()
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        e.printStackTrace()
        emptyList()
    }
}