package org.yazan;

import basic.Demo.saUtils;
import basic.bikeCADPro;
import basic.myUtils;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Locale;
import java.util.ResourceBundle;

public class BikeCadWrapper {
    private static final Locale LOCALE = Locale.forLanguageTag("en-us");
    private static final ResourceBundle RESOURCE_BUNDLE = ResourceBundle.getBundle("Language", LOCALE);
    private static final String CONFIG_DIR = saUtils.getConfigPointer() + "BikeCAD_" + myUtils.Version() + "_configuration" + File.separatorChar;

    public static bikeCADPro create() {
        try {
            return initBikeCad();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initiate BikeCad", e);
        }
    }


    private static bikeCADPro initBikeCad() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        bikeCADPro bikeCADPro = new bikeCADPro();
        setStaticField(bikeCADPro, "myLocale", LOCALE);
        setStaticField(bikeCADPro, "res", RESOURCE_BUNDLE);
        setStaticField(bikeCADPro, "confDir", CONFIG_DIR);
        invokeVoidMethod(bikeCADPro, "splashScreenInit");
        bikeCADPro.init();
        return bikeCADPro;
    }

    private static void invokeVoidMethod(bikeCADPro bikeCADPro, String methodName) throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        Method method = bikeCADPro.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        method.invoke(bikeCADPro);
    }

    private static void setStaticField(bikeCADPro bikeCADPro,
                                       String fieldName,
                                       Object value) throws NoSuchFieldException, IllegalAccessException {
        Field field = bikeCADPro.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(null, value);
    }

}
