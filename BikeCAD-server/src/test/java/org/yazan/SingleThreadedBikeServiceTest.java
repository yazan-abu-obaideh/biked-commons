package org.yazan;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
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
        URL bcadResource = getNonNullResource(resourceHandle);
        URL imageResource = getNonNullResource(resourceHandle.replace(".bcad", ".svg"));
        byte[] bikeRenderingResult = BIKE_SERVICE.renderBike(Files.readString(Path.of(bcadResource.getPath())));
        assertNotEquals(0, bikeRenderingResult.length);
        assertTrue(similarArrays(Files.readAllBytes(Path.of(imageResource.getPath())), bikeRenderingResult));
    }

    private boolean similarArrays(byte[] first, byte[] second) {
        if (first.length == 0 && second.length == 0) {
            return true;
        }
        int smallerLength = Math.min(first.length, second.length);
        int lengthDifference = Math.abs(first.length - second.length);
        var fractionLengthDifference = BigDecimal.valueOf(lengthDifference)
                .divide(BigDecimal.valueOf(smallerLength), RoundingMode.HALF_UP);
        System.out.println("fractionLengthDifference: " + fractionLengthDifference);
        if (fractionLengthDifference.compareTo(BigDecimal.valueOf(0.000_01)) > 0) {
            return false;
        }
        int numberDifferent = 0;
        for (int i = 0; i < smallerLength; i++) {
            if (first[i] != second[i]) {
                numberDifferent += 1;
            }
        }
        BigDecimal fractionDifference = BigDecimal.valueOf(numberDifferent).divide(BigDecimal.valueOf(smallerLength), RoundingMode.HALF_UP);
        System.out.println("fractionDifferent: " + fractionDifference);
        return fractionDifference.compareTo(BigDecimal.valueOf(0.000_01)) < 0;
    }

    static List<Integer> bikeIndices() {
        ArrayList<Integer> indices = new ArrayList<>();
        for (int i = 1; i <= 13; i++) {
            indices.add(i);
        }
        return indices;
    }

    static URL getNonNullResource(String resourceHandle) {
        return Objects.requireNonNull(SingleThreadedBikeServiceTest.class.getClassLoader().getResource(resourceHandle), "Could not find resource " + resourceHandle);
    }

}