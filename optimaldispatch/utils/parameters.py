# Exports
__all__ = ["Parameters"]


class Parameters(dict):
    """
    Class used to keep parameters. It extends the dict type to provide a get
    method that, besides allows the user to define a default value for
    non-existent keys, it allows the user to define a function that is used to
    process the value before returning. By default, the function is not called
    for default value.
    """

    def get(self, key, default=None, converter=None, convert_default=False):
        """
        Get an attribute by the key.

        :param key: A key.
        :param default: A default value to return if the key is not present.
        :param converter: A function used to process the value bonded to the
               key. If set, this method returns the result of this function
               instead the value itself. This function must receive a single
               parameter.
        :param convert_default: If set to True, the converter function will
               process the default value if the key is not present. Otherwise,
               the default value is returned as it is.
        :return: A value bonded to the key.
        """

        if key in self:
            if self[key] is not None and converter is not None:
                return converter(self[key])
            else:
                return self[key]
        else:
            if default is not None and converter is not None and convert_default is True:
                return converter(default)
            else:
                return default
