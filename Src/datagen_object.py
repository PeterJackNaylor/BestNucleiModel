

from segmentation_net import ExampleUNetDatagen, ExampleDistDG

def return_range(inti):
    if inti == 0:
        min_, max_ = 0, 255
    elif inti == 1:
        min_, max_ = 250, 500
    elif inti == 2:
        min_, max_ = 500, 750
    elif inti == 3:
        min_, max_ = 750, 1000
    return min_, max_


class DataGenNeerajBin(ExampleUNetDatagen):
    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2

        row = integer % (self.crop**0.5)
        col = integer // (self.crop**0.5)

        x_b, x_e = return_range(row)
        y_b, y_e = return_range(row)
        
        x_e += add2
        y_e += add2

        return image[x_b:x_e, y_b:y_e]


class DataGenNeeraj(ExampleDistDG):
    def quarter_image(self, image, integer):
        """
        Split images in quarters dependings of integer
        with additionnal expanding
        """
        add2 = self.add * 2

        row = integer % (self.crop**0.5)
        col = integer // (self.crop**0.5)

        x_b, x_e = return_range(row)
        y_b, y_e = return_range(row)
        
        x_e += add2
        y_e += add2

        return image[x_b:x_e, y_b:y_e]