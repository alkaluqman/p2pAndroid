package com.example.applayout.Data.Database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.example.applayout.Data.Database.Daos.LocalModelDao
import com.example.applayout.Data.Model.LocalModel

@Database(entities = [LocalModel::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun localModelDao(): LocalModelDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "room_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}