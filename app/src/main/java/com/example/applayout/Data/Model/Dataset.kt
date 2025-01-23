package com.example.applayout.Data.Model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "LocalDatasets")
data class Dataset(
    @PrimaryKey val uniqueIdentifier: String,
    var model_task: String = "object_detection",
    var description: String = "",
    var numImages: Int = 0,
    var class_labels: String = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck"
) {
    fun getClassLabelsAsList(): List<String> {
        return class_labels.split(",").map { it.trim() }
    }
}



