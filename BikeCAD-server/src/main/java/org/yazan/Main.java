package org.yazan;

import basic.bikeCADPro;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Main {

    // TODO: find out why it's considered 'headless' if it's instantiated elsewhere
    public static final bikeCADPro BIKE_CAD_INSTANCE = BikeCadWrapper.create();

    public static void main(String[] args) {
        SpringApplication.run(Main.class, args);
    }
}