package org.yazan;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/render")
public class RenderingController {
    @Autowired
    private BikeService bikeService;

    @PostMapping
    public byte[] renderBike(@RequestBody String bikeXml) {
        System.out.println("Receiving rendering request...");
        return bikeService.renderBike(bikeXml);
    }
}
