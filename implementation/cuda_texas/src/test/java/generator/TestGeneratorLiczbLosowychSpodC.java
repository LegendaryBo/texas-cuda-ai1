package generator;

import java.util.Arrays;

import junit.framework.TestCase;

public class TestGeneratorLiczbLosowychSpodC extends TestCase {

    GeneratorLiczbLosowychSpodC generator = new GeneratorLiczbLosowychSpodC();
    
    public void testLiczbyDodatnie() {
        final int LICZBA_SPRAWDZEN=5000;
        
        int[] pIntTable = new int[LICZBA_SPRAWDZEN];
        int powtorzenia=0;
        for (int i=0; i < LICZBA_SPRAWDZEN; i++) {
            int wylosowana = generator.nextInt();
            if (Arrays.binarySearch(pIntTable, wylosowana) > 0)
                powtorzenia++;
            pIntTable[i] = wylosowana;
            assertTrue(wylosowana>=0);
        }
        assertTrue(powtorzenia<5);
        System.out.println(powtorzenia);
    }
    
    public void testTakieSameLiczby() {
        final int LICZBA_SPRAWDZEN=5000;
        GeneratorLiczbLosowychSpodC generator2 = new GeneratorLiczbLosowychSpodC();
  
        for (int i=0; i < LICZBA_SPRAWDZEN; i++) {
            int wylosowana2 = generator.nextInt();
            int wylosowana = generator2.nextInt();
            assertEquals(wylosowana, wylosowana2);
        }

    }
    
}
