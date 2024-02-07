__kernel void negative(__global char* p)
{
	int gid = get_global_id(0);
	p[gid] = 255 - p[gid];
}

typedef struct {
	uchar r;
	uchar g;
	uchar b;
}Pixel;

__kernel void gaussianBlur(__global Pixel* p, __global float* core, __global int* args)
{
	int gid = get_global_id(0);

	int width = args[0], height = args[1], radius = args[2], coreSize = args[3];

	int x = gid % width;
	int y = gid / width;

	float r = 0.0, g = 0.0, b = 0.0;

	for (int i = -radius; i <= radius; i++)
	{
		int deltaX = 0;
		if (x + i < 0 || x + i >= width) deltaX = -i;
		else deltaX = i;

		for (int j = -radius; j <= radius; j++)
		{
			int deltaY = 0;
			if (y + j < 0 || y + j >= height) deltaY = -j;
			else deltaY = j;

			int index = ((y + deltaY) * width + x + deltaX);
			r += convert_float(p[index].r) * core[(i + radius) * coreSize + j + radius];
			g += convert_float(p[index].g) * core[(i + radius) * coreSize + j + radius];
			b += convert_float(p[index].b) * core[(i + radius) * coreSize + j + radius];
		}
	}

	p[gid].r = convert_uchar_rte(r);
	p[gid].g = convert_uchar_rte(g);
	p[gid].b = convert_uchar_rte(b);
}