package com.example.applayout.Data.Model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "LocalDatasets")
data class Dataset(
    @PrimaryKey val uniqueIdentifier: String,
    var model_task: String = "object_detection",
    var description: String = "",
    var class_labels: String = "T-shirt/top,Trouser,Pullover,Dress,Coat,Sandal,Shirt,Sneaker,Bag,Ankle boot",
    var isUploaded: Boolean = false,
    var absoluteFilePath: String = ""
) {
    fun getClassLabelsAsList(): List<String> {
        return class_labels.split(",").map { it.trim() }
    }
}



