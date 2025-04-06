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

    private static bikeCADPro INSTANCE = null;

    public static synchronized bikeCADPro create() {
        if (INSTANCE == null) {
            init();
        }
        return INSTANCE;
    }

    private static void init() {
        try {
            INSTANCE = initBikeCad();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initiate BikeCad", e);
        }
    }


    private static bikeCADPro initBikeCad() throws NoSuchFieldException, IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        bikeCADPro instance = new bikeCADPro();
        setStaticField(bikeCADPro.class, "myLocale", LOCALE);
        setStaticField(bikeCADPro.class, "res", RESOURCE_BUNDLE);
        setStaticField(bikeCADPro.class, "confDir", CONFIG_DIR);
        invokeVoidMethod(instance, "splashScreenInit");
        instance.init();
        return instance;
    }

    private static void invokeVoidMethod(bikeCADPro bikeCADPro, String methodName) throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        Method method = bikeCADPro.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        method.invoke(bikeCADPro);
    }

    private static void setStaticField(Class<?> aClass,
                                       String fieldName,
                                       Object value) throws NoSuchFieldException, IllegalAccessException {
        Field field = aClass.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(null, value);
    }

}
