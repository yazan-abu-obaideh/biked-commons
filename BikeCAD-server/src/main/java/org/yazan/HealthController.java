package org.yazan;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/actuator")
public class HealthController {
    @GetMapping("/serverInformation")
    public ServerInformation getInformation() {
        return new ServerInformation("BikeCAD-server", "UP");
    }

    public record ServerInformation(String serverName, String status) {

    }
}
