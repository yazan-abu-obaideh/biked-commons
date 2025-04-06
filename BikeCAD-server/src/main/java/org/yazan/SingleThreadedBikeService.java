package org.yazan;

import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static org.yazan.Main.BIKE_CAD_INSTANCE;

@Service
public class SingleThreadedBikeService implements BikeService {

    public static final String BIKE_CAD_FILE_EXTENSION = ".bcad";

    /**
     * Synchronized because BikeCAD internally uses static variables and some such for state management.
     * Concurrent calls will result in mangled bikes.
     */
    @Override
    public synchronized byte[] renderBike(String bikeXml) {
        System.out.println("Service accepted rendering request...");
        try {
            Path tempDirectory = Files.createTempDirectory(UUID.randomUUID().toString());
            String bikeFileBase = UUID.randomUUID().toString();
            Path tempBikeFile = tempDirectory.resolve(bikeFileBase.concat(BIKE_CAD_FILE_EXTENSION));
            Files.writeString(tempBikeFile, bikeXml);
            BIKE_CAD_INSTANCE.BatchOperation(1, tempDirectory.toFile());
            Path imageFile = tempBikeFile.resolveSibling(bikeFileBase.concat(".svg"));
            byte[] result = Files.readAllBytes(imageFile);
            System.out.println("Service processed rendering request...");
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
