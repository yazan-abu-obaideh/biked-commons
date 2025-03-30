package org.yazan;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class SingleThreadedBikeServiceTest {

    public static final SingleThreadedBikeService BIKE_SERVICE = new SingleThreadedBikeService();

    @ParameterizedTest
    @MethodSource("bikeIndices")
    void renderBike(int bikeIndex) throws IOException {
        String resourceHandle = "bikes/bike%s.bcad".formatted(bikeIndex);

        byte[] bikeRenderingResult = BIKE_SERVICE.renderBike(getBikeXml(resourceHandle));
        assertNotEquals(0, bikeRenderingResult.length);

        byte[] expectedBytes = getExpectedImage(resourceHandle);
        assertArrayEquals(expectedBytes, bikeRenderingResult);
    }

    private static String getBikeXml(String resourceHandle) throws IOException {
        URL bcadResource = getNonNullResource(resourceHandle);
        return Files.readString(Path.of(bcadResource.getPath()));
    }

    private static byte[] getExpectedImage(String resourceHandle) throws IOException {
        URL imageResource = getNonNullResource(resourceHandle.replace(".bcad", ".svg"));
        return Files.readAllBytes(Path.of(imageResource.getPath()));
    }

    static List<Integer> bikeIndices() {
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 1; i <= 13; i++) {
            indices.add(i);
        }
        return indices;
    }

    static URL getNonNullResource(String resourceHandle) {
        return Objects.requireNonNull(SingleThreadedBikeServiceTest.class.getClassLoader().getResource(resourceHandle),
                "Could not find resource " + resourceHandle);
    }

}