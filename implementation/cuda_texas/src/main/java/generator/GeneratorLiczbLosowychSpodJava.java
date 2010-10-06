package generator;

import java.util.Random;

public class GeneratorLiczbLosowychSpodJava extends Random {

	private int m_z=7542;
	private int m_w=92465;

	public int nextInt() {

		m_z = 36969 * (m_z & 65535) + (m_z >> 16);
		m_w = 18000 * (m_w & 65535) + (m_w >> 16);
		int wynik = (m_z << 16) + (m_w & 65535);
		if (wynik < 0)
			return -wynik;
		else
			return wynik;

	}

}
