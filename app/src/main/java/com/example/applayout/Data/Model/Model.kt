package com.example.applayout.Data.Model


data class Model(
    var uniqueIdentifier: String = "",
    var model_task: String = "object_detection",
    var last_trained: String = "",
    var description: String = "",
    var weight_size: Long = 0,
    var is_uploaded: Boolean = false,
    var usage: Int = 0,
    var likes: Int = 0,
    var public_link: String = "",
    var architecture: String = "",
    var isOwner: Boolean = false,
    var absoluteFilePath: String = ""

) {
    fun toLocalModel(): LocalModel {
        return LocalModel(
            uniqueIdentifier = this.uniqueIdentifier,
            model_task = this.model_task,
            description = this.description,
            absoluteFilePath = this.absoluteFilePath
        )
    }
}
